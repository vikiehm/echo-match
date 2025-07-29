import os
import os.path as osp
import numpy as np
import torch
import inspect
import scipy.sparse as sp  # added to support sparse matrices
from utils.geometry_util import torch2np, hash_arrays

def get_cached_compute(compute_fn, verts, faces, cache_dir=None, **kwargs):
    """
    Generic wrapper for computing and caching shape properties.
    Wraps any compute function that takes verts and faces as input, using a cache if possible.
    Supports compute functions that return one or multiple results.
    All arrays are always computed using double precision for stability,
    then truncated to single precision floats to store on disk,
    and finally returned as torch tensors with dtype/device matching the `verts` input.
    This version also accommodates both scipy sparse matrices and torch sparse tensors.
    
    Args:
        compute_fn: Function that computes the desired property from verts and faces.
                    It can return a single result (tensor/ndarray/sparse matrix) or multiple results (tuple or list).
        verts: Vertex positions tensor.
        faces: Face indices tensor.
        cache_dir: Directory to cache results.
        **kwargs: Additional arguments passed to compute_fn.
    """
    assert verts.dim() == 2  # we don't consider batch
    device = verts.device
    dtype = verts.dtype
    verts_np = torch2np(verts.float())  # use 32-bit for hash search and in production
    faces_np = torch2np(faces) if faces is not None else None

    if np.isnan(verts_np).any():
        raise ValueError('detect NaN vertices.')

    # Helper to serialize a computed result.
    def serialize_result(x):
        if torch.is_tensor(x):
            if x.is_sparse:
                x = x.coalesce()
                indices = x._indices().cpu().numpy().astype(np.int32)
                values = x._values().cpu().numpy().astype(np.float32)
                shape = np.array(x.size(), dtype=np.int32)
                return {
                    '__sparse__': True,
                    'framework': 'torch',
                    'indices': indices,
                    'values': values,
                    'shape': shape
                }
            else:
                return torch2np(x)
        elif sp.issparse(x):
            x_coo = x.tocoo()
            return {
                '__sparse__': True,
                'framework': 'scipy',
                'data': x_coo.data.astype(np.float32),
                'row': x_coo.row.astype(np.int32),
                'col': x_coo.col.astype(np.int32),
                'shape': np.array(x_coo.shape, dtype=np.int32)
            }
        else:
            return x

    # Helper to deserialize a saved result.
    def deserialize_result(x):
        # If loaded object was pickled, extract the original Python dict.
        if isinstance(x, np.ndarray) and x.dtype == np.object_:
            x = x.item()
        if isinstance(x, dict) and x.get('__sparse__', False):
            if x.get('framework', 'scipy') == 'torch':
                indices = torch.from_numpy(x['indices']).long().to(device=device)
                values = torch.from_numpy(x['values']).to(device=device, dtype=dtype)
                shape = tuple(x['shape'])
                return torch.sparse_coo_tensor(indices, values, size=shape, device=device, dtype=dtype)
            else:
                coo = sp.coo_matrix((x['data'], (x['row'], x['col'])), shape=tuple(x['shape']))
                indices = np.vstack((coo.row, coo.col))
                indices = torch.from_numpy(indices).long().to(device=device)
                values = torch.from_numpy(coo.data).to(device=device, dtype=dtype)
                return torch.sparse_coo_tensor(indices, values, size=coo.shape, device=device, dtype=dtype)
        else:
            return torch.from_numpy(x).to(device=device, dtype=dtype)

    found = False
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        hash_key_str = str(hash_arrays((verts_np, faces_np)))
        i_cache = 0
        while True:
            search_path = osp.join(cache_dir, hash_key_str + '_' + str(i_cache) + '.npz')
            try:
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile['verts']
                cache_faces = npzfile['faces']
                # Check if cached verts and faces match.
                if (not np.array_equal(verts_np, cache_verts)) or (not np.array_equal(faces_np, cache_faces)):
                    i_cache += 1
                    print('collision detected')
                    continue

                # Load cached result(s)
                if 'result_count' in npzfile:
                    count = int(npzfile['result_count'])
                    results = []
                    for i in range(count):
                        res_serialized = npzfile[f'result_{i}']
                        results.append(deserialize_result(res_serialized))
                    result = tuple(results)
                else:
                    res_serialized = npzfile['result']
                    result = deserialize_result(res_serialized)
                found = True
                break
            except FileNotFoundError:
                # Cache file not found; we will compute the value.
                break
            except Exception:
                # If any other exception occurs (e.g., corrupt file), remove it and break.
                os.remove(search_path)
                break

    if not found:
        # Compute the result.
        print(f"cache miss, computing {compute_fn.__name__}")
        sig = inspect.signature(compute_fn)
        if 'faces' in sig.parameters:
            try:
                result = compute_fn(verts, faces, **kwargs)
            except Exception:
                result = compute_fn(verts_np, faces_np, **kwargs)
        else:
            try:
                result = compute_fn(verts, **kwargs)
            except Exception:
                result = compute_fn(verts_np, **kwargs)

        # Convert results to a serializable format.
        if isinstance(result, (tuple, list)):
            result_converted = tuple(serialize_result(r) for r in result)
        else:
            result_converted = serialize_result(result)

        dtype_np = np.float32

        if cache_dir:
            # Save computed result(s) in the cache.
            if isinstance(result_converted, tuple):
                save_dict = {
                    'verts': verts_np,
                    'faces': faces_np,
                    'result_count': np.array(len(result_converted), dtype=np.int32)
                }
                for i, res in enumerate(result_converted):
                    if isinstance(res, dict):
                        save_dict[f'result_{i}'] = res
                    else:
                        save_dict[f'result_{i}'] = res.astype(dtype_np)
            else:
                if isinstance(result_converted, dict):
                    save_dict = {
                        'verts': verts_np,
                        'faces': faces_np,
                        'result': result_converted
                    }
                else:
                    save_dict = {
                        'verts': verts_np,
                        'faces': faces_np,
                        'result': result_converted.astype(dtype_np)
                    }
            np.savez(search_path, **save_dict)

            # Convert saved result(s) back to torch or sparse tensors.
            if isinstance(result_converted, tuple):
                result = tuple(deserialize_result(save_dict[f'result_{i}'])
                               for i in range(len(result_converted)))
            else:
                result = deserialize_result(save_dict['result'])
        else:
            # No caching; directly convert result(s) to torch or sparse tensors.
            if isinstance(result_converted, tuple):
                result = tuple(
                    deserialize_result(r) if isinstance(r, dict) else deserialize_result(r.astype(dtype_np))
                    for r in result_converted
                )
            else:
                result = deserialize_result(result_converted if isinstance(result_converted, dict)
                                            else result_converted.astype(dtype_np))
    return result