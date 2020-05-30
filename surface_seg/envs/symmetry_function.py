from simple_nn.features.symmetry_function._libsymf import lib, ffi
import numpy as np
import copy

# This file was adopted from SimpleNN's excellent fingerprinting code.

def _gen_2Darray_for_ffi(arr, ffi, cdata="double"):
    # Function to generate 2D pointer for cffi
    shape = arr.shape
    arr_p = ffi.new(cdata + " *[%d]" % shape[0])
    for i in range(shape[0]):
        arr_p[i] = ffi.cast(cdata + " *", arr[i].ctypes.data)
    return arr_p

def wrap_symmetry_functions(atoms, params_set):
    # Adapted from the python code in simple-nn

    x_out = {}
    dx_out = {}
    da_out = {}
    
    cart = np.copy(atoms.get_positions(wrap=True), order='C')
    scale = np.copy(atoms.get_scaled_positions(), order='C')
    cell = np.copy(atoms.cell, order='C')

    symbols = np.array(atoms.get_chemical_symbols())
    atom_num = len(symbols)
    atom_i = np.zeros([len(symbols)], dtype=np.intc, order='C')
    type_num = dict()
    type_idx = dict()
    
    for j,jtem in enumerate(params_set.keys()):
        tmp = symbols==jtem
        atom_i[tmp] = j+1
        type_num[jtem] = np.sum(tmp).astype(np.int64)
        # if atom indexs are sorted by atom type,
        # indexs are sorted in this part.
        # if not, it could generate bug in training process for force training
        type_idx[jtem] = np.arange(atom_num)[tmp]

    for key in params_set:
        params_set[key]['ip']=_gen_2Darray_for_ffi(np.asarray(params_set[key]['i'], dtype=np.intc, order='C'), ffi, "int")
        params_set[key]['dp']=_gen_2Darray_for_ffi(np.asarray(params_set[key]['d'], dtype=np.float64, order='C'), ffi)
        
    atom_i_p = ffi.cast("int *", atom_i.ctypes.data)

    cart_p  = _gen_2Darray_for_ffi(cart, ffi)
    scale_p = _gen_2Darray_for_ffi(scale, ffi)
    cell_p  = _gen_2Darray_for_ffi(cell, ffi)

    for j,jtem in enumerate(params_set.keys()):
        q = type_num[jtem]
        r = type_num[jtem] 

        cal_atoms = np.asarray(type_idx[jtem][:], dtype=np.intc, order='C')
        cal_num = len(cal_atoms)
        cal_atoms_p = ffi.cast("int *", cal_atoms.ctypes.data)

        x = np.zeros([cal_num, params_set[jtem]['num']], dtype=np.float64, order='C')
        dx = np.zeros([cal_num, params_set[jtem]['num'] * atom_num * 3], dtype=np.float64, order='C')
        da = np.zeros([cal_num, params_set[jtem]['num'] * 3 * 6], dtype=np.float64, order='C')

        x_p = _gen_2Darray_for_ffi(x, ffi)
        dx_p = _gen_2Darray_for_ffi(dx, ffi)
        da_p = _gen_2Darray_for_ffi(da, ffi)

        errno = lib.calculate_sf(cell_p, cart_p, scale_p, \
                         atom_i_p, atom_num, cal_atoms_p, cal_num, \
                         params_set[jtem]['ip'], params_set[jtem]['dp'], params_set[jtem]['num'], \
                         x_p, dx_p, da_p)
                
        x_out[jtem] = np.array(x)
        dx_out[jtem] = np.array(dx)
        da_out[jtem] = np.array(da)
        
    # re-arrange x_out using atom_index 

    all_atom_indices = np.concatenate([type_idx[a] for a in type_idx])
    all_fingerprints = np.concatenate([x_out[a] for a in x_out])
    
    sorted_all_fingerprints = all_fingerprints[all_atom_indices,:]

    return sorted_all_fingerprints

#Adapted from the amptorch make_snn_params function
def make_snn_params(
    elements, etas, rs_s, g4_eta=4, cutoff=6.5, g4_zeta=[1.0, 4.0], g4_gamma=[1, -1]
    ):
    """
    makes a params file for simple_NN. This is the file containing
    the descriptors. This function makes g2 descriptos for the eta
    and rs values that are input, and g4 descriptors that are log
    spaced between 10 ** -5 and 10 ** -1. The number of these
    that are made is controlled by the `n_g4_eta` variable
    Parameters:
        elements (list):
            a list of elements for which you'd like to make params
            files for
        etas (list):
            the eta values you'd like to use for the descriptors
        rs_s (list):
            a list corresponding to `etas` that contains the rs
            values for each descriptor
        g4_eta (int or list):
            the number of g4 descriptors you'd like to use. if a
            list is passed in the values of the list will be used
            as eta values
        cutoff (float):
            the distance in angstroms at which you'd like to cut
            off the descriptors
    returns:
        None
    """
    
    params_set = {}
    
    if len(etas) != len(rs_s):
        raise ValueError('the length of the etas list must be equal to the'
                         'length of the rs_s list')
    if type(g4_eta) == int:
        g4_eta = np.logspace(-4, -1, num=g4_eta)
    for element in elements:
        params = {'i':[],'d':[]}
        
        # G2
        for eta, Rs in zip(etas, rs_s):
            for species in range(1, len(elements) + 1):
                params['i'].append([2,species,0])
                params['d'].append([cutoff,eta,Rs,0.0])

        # G4
        for eta in g4_eta:
            for zeta in g4_zeta:
                for lamda in g4_gamma:
                    for i in range(1, len(elements) + 1):
                        for j in range(i, len(elements) + 1):
                            params['i'].append([4,i,j])
                            params['d'].append([cutoff,eta,zeta,lamda])
                            
                            
        params_set[element]={'num':len(params['i']),
                'i':params['i'],
                'd':params['d']}
    return params_set