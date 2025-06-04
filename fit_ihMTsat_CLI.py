# ///////////////////////////////////////////////////////////////////////////////////////////////
# // L. SOUSTELLE, PhD, Aix Marseille Univ, CNRS, CRMBM, Marseille, France
# // 2020/12/27
# // Contact: lucas.soustelle@univ-amu.fr
# ///////////////////////////////////////////////////////////////////////////////////////////////

import argparse
import multiprocessing
import os
import subprocess
import sys
import time
from argparse import RawTextHelpFormatter

import nibabel as nb
import numpy as np
import scipy.integrate
import scipy.linalg
import scipy.optimize


class OptionalFileAction(argparse.Action):
    """Custom action for optional file arguments that can have default filenames."""

    def __init__(self, option_strings, dest, default_filename=None, **kwargs):
        self.default_filename = default_filename
        # Set nargs to '?' to allow optional argument
        kwargs['nargs'] = '?'
        # Set const to a sentinel value to detect when flag is used without argument
        kwargs['const'] = '__FLAG_WITHOUT_VALUE__'
        # Set default to None so unspecified flags result in None
        kwargs['default'] = None
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values == '__FLAG_WITHOUT_VALUE__':
            # Flag was used without a value, use default filename
            setattr(namespace, self.dest, self.default_filename)
        else:
            # Flag was used with a value, or not used at all (None)
            setattr(namespace, self.dest, values)


class SplitCommaAction(argparse.Action):
    """Custom action to split comma-separated values into a list of ints."""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, [int(v) - 1 for v in values.split(',')])


def _get_parser():
    from functools import partial

    text_description = (
        'Compute ihMT and MT saturation maps [1] from an ihMT-prepared experiment [2]. '
        'Outputs are in percentage unit.'
        '\nNotes:'
        '\n\t1) The estimation assumes a centric-out readout, in steady-state.'
        '\n\t2) Providing a B1 map is strongly recommended.'
        '\n\t3) The number of FLASH repetition in TFL should encompass acceleration settings '
        '(e.g., GRAPPA).'
        '\n\t4) Volume indices (see --R/--S/--D options) start at 1.'
        '\n\t5) To Siemens users using the C2P ihMT-RAGE sequence '
        '(Aix Marseille Univ, CNRS, CRMBM):'
        '\n\t - The last BTR duration can be calculated as: '
        '\n\t\t BTR_last = [ihMT Prep. time] - [# of bursts - 1]*[Burst TR].'
        '\n\t - The TFL "FLASH TR" is equal to the [Echo Spacing] variable in Sequence/Part 1.'
        '\n\t - The TFL "FLASH number of repetition" is related to the [Turbo Factor] (TF) in '
        'Sequence/Part 1:'
        '\n\t\t - Linear encoding with integrated/separated GRAPPA: TF'
        '\n\t\t - Linear Rot. encoding with integrated GRAPPA: floor(TF/2)-1 + #ACS/2'
        '\n\t\t - Linear Rot. encoding with separated GRAPPA: floor(TF/2)-1'
        '\nReferences:'
        '\n\t [1] G. Helms et al., High-resolution maps of magnetization transfer with inherent'
        'correction for RF inhomogeneity and T1 relaxation obtained from 3D FLASH MRI, '
        'MRM 2008;60:1396-1407'
        '\n\t [2] F. Munsch et al., Characterization of the cortical myeloarchitecture '
        'with inhomogeneous magnetization transfer imaging (ihMT), NeuroImage 2021;225:117442'
    )
    parser = argparse.ArgumentParser(
        description=text_description,
        formatter_class=RawTextHelpFormatter,
    )
    IsFile = partial(is_valid_file, parser=parser)

    parser.add_argument(
        'ihmt',
        type=IsFile,
        help='Input 4D NIfTI path',
    )
    parser.add_argument(
        't1',
        type=IsFile,
        help='Input T1 (in sec) NIfTI path',
    )
    parser.add_argument(
        '--ihmtsat',
        dest='ihmtsat_file',
        action='store',
        default='ihMTsat.nii.gz',
        help=(
            'Output ihMTsat NIfTI path. '
            'This will always be created, even if not specified. '
            'Default: ihMTsat.nii.gz'
        ),
    )
    parser.add_argument(
        'ihmtparx',
        nargs='?',
        help="""ihMT preparation parameters (comma-separated), in this order:
        *** ihMT-RAGE:
        1) Interpulse delay ("delta_t"; ms)
        2) Number of pulses per burst (int)
        3) Total number of bursts (int)
        4) Burst TR (BTR; ms)
        5) Last Burst TR (ms)
        e.g. 0.8,10,5,100.0,10.0
        *** ihMT-GRE:
        1) Interpulse delay ("delta_t"; ms)
        2) Number of pulses per preparation (int)
        3) Post-ihMT preparation delay (ms)
        e.g. 0.8,10,2.0""",
    )
    parser.add_argument(
        'tflparx',
        help="""TurboFLASH readout and sequence parameters (comma-separated), in this order:
        1) FLASH TR (ms)
        2) Readout flip angle (deg)
        3) FLASH number of repetition (int)
        4) Sequence Time to Repetition (TR; ms)
        e.g. 8.3,7,128,4000.0""",
    )
    parser.add_argument(
        '--mtdsat',
        dest='mtdsat_file',
        action=OptionalFileAction,
        default_filename='MTdsat.nii.gz',
        help=(
            'Output MTsat from dual-offset images NIfTI path. '
            'This will only be created if specified. '
            'Providing the flag without a value will use the default filename.'
        ),
    )
    parser.add_argument(
        '--mtssat',
        dest='mtssat_file',
        action=OptionalFileAction,
        default_filename='MTssat.nii.gz',
        help=(
            'Output MTsat from single-offset images NIfTI path. '
            'This will only be created if specified. '
            'Providing the flag without a value will use the default filename.'
        ),
    )
    parser.add_argument(
        '--ihmtsatb1sq',
        dest='ihmtsatb1sq_file',
        action=OptionalFileAction,
        default_filename='ihMTsatB1sq.nii.gz',
        help=(
            'Output ihMTsat image normalized by squared B1 NIfTI path. '
            'This will only be created if specified and a B1 map is provided. '
            'Providing the flag without a value will use the default filename.'
        ),
    )
    parser.add_argument(
        '--mtdsatb1sq',
        dest='mtdsatb1sq_file',
        action=OptionalFileAction,
        default_filename='MTdsatB1sq.nii.gz',
        help=(
            'Output MTsat from single-offset images normalized by squared B1 NIfTI path. '
            'This will only be created if specified and a B1 map is provided. '
            'Providing the flag without a value will use the default filename.'
        ),
    )
    parser.add_argument(
        '--mtssatb1sq',
        dest='mtssatb1sq_file',
        action=OptionalFileAction,
        default_filename='MTssatB1sq.nii.gz',
        help=(
            'Output MTsat from dual-offset images normalized by squared B1 NIfTI path. '
            'This will only be created if specified and a B1 map is provided. '
            'Providing the flag without a value will use the default filename.'
        ),
    )
    parser.add_argument(
        '--b1',
        type=IsFile,
        help='Input B1 NIfTI (in absolute units).',
    )
    parser.add_argument(
        '--mask',
        type=IsFile,
        help='Input binary mask NIfTI path',
    )
    parser.add_argument(
        '--reference-idx',
        action=SplitCommaAction,
        default=0,
        help=(
            'Reference indices (starting from 1) in 4D input '
            '(comma-separated integers; default: 1)'
        ),
    )
    parser.add_argument(
        '--single-offset-idx',
        action=SplitCommaAction,
        help=(
            'Single-offset indices (starting from 1) in 4D input '
            '(comma-separated integers; default: 2,4,...,N-1)'
        ),
    )
    parser.add_argument(
        '--dual-offset-idx',
        action=SplitCommaAction,
        help=(
            'Dual-offset indices (starting from 1) in 4D input '
            '(comma-separated integers; default: 3,5,...,N)'
        ),
    )
    parser.add_argument(
        '--xtol',
        type=float,
        default=1e-6,
        help='x tolerance for root finding (default: 1e-6)',
    )
    parser.add_argument(
        '--nworkers',
        type=int,
        default=1,
        help='Use this for multi-threading computation (default: 1)',
    )
    return parser


def main(
    *,
    ihmt,
    t1,
    ihmtsat_file,
    ihmtparx,
    tflparx,
    mtdsat_file,
    mtssat_file,
    ihmtsatb1sq_file,
    mtdsatb1sq_file,
    mtssatb1sq_file,
    b1,
    mask,
    reference_idx,
    single_offset_idx,
    dual_offset_idx,
    xtol,
    nworkers,
):
    nworkers = nworkers if nworkers <= get_cpu_count() else get_cpu_count()

    #### Sequence parx
    print('')
    print('--------------------------------------------------')
    print('----- Checking entries for ihMTsat processing ----')
    print('--------------------------------------------------')
    print('')

    ihmtparx = ihmtparx.split(',')
    if len(ihmtparx) == 5:
        exp_rage_gre = 'RAGE'  ## RAGE experiment
        ihmtparx = {
            'dt': float(ihmtparx[0]) * 1e-3,
            'Np': int(ihmtparx[1]),
            'NBTR': int(ihmtparx[2]),
            'BTR': float(ihmtparx[3]) * 1e-3,
            'BTRlast': float(ihmtparx[4]) * 1e-3,
        }
        print('Summary of input ihMT-RAGE preparation parameters:')
        print(f'\t Delta_t: {ihmtparx["dt"] * 1e3:.1f} ms')
        print(f'\t Number of pulses per burst: {ihmtparx["Np"]}')
        print(f'\t Number of bursts: {ihmtparx["NBTR"]}')
        print(f'\t Burst TR: {ihmtparx["BTR"] * 1e3:.1f} ms')
        print(f'\t Last Burst TR: {ihmtparx["BTRlast"] * 1e3:.1f} ms')
        print('')
    elif len(ihmtparx) == 3:
        exp_rage_gre = 'GRE'
        ihmtparx = {
            'dt': float(ihmtparx[0]) * 1e-3,
            'Np': int(ihmtparx[1]),
            'pihMTdelay': float(ihmtparx[2]) * 1e-3,
        }
        print('Summary of input ihMT-GRE preparation parameters:')
        print(f'\t Delta_t: {ihmtparx["dt"] * 1e3:.1f} ms')
        print(f'\t Number of pulses per preparation: {ihmtparx["Np"]}')
        print(f'\t Post-ihMT preparation delay: {ihmtparx["pihMTdelay"] * 1e3}')
        print('')
    else:
        raise ValueError(
            'Wrong amount of ihMT preparation parameters '
            f'(ihmtparx --- expected 3 (ihMT-GRE) or 5 (ihMT-RAGE), found {len(ihmtparx)})'
        )

    tflparx = tflparx.split(',')
    if len(tflparx) != 4:
        raise ValueError(
            'Wrong amount of sequence/readout parameters '
            f'(tflparx --- expected 4, found {len(tflparx)})'
        )
    tflparx = {
        'subTR': float(tflparx[0]) * 1e-3,
        'FA': float(tflparx[1]),
        'Npart': int(tflparx[2]),
        'TR': float(tflparx[3]) * 1e-3,
    }
    print('Summary of TFL parameters:')
    print(f'\t subTR: {tflparx["subTR"] * 1e3:.1f} ms')
    print(f'\t Readout FA: {tflparx["FA"]:.1f} deg')
    print(f'\t Number of TFL rep.: {tflparx["Npart"]}')
    print(f'\t Sequence TR: {tflparx["TR"] * 1e3:.1f} ms')
    print('')

    # last check
    for k, v in ihmtparx.items():
        if v < 0:
            raise ValueError(f'All ihmtparx values should be positive: {k} = {v}')
    for k, v in tflparx.items():
        if v < 0:
            raise ValueError(f'All tflparx values should be positive: {k} = {v}')

    #### check input data
    # check ihMT data
    if nb.load(ihmt).ndim != 4:
        raise ValueError(f'Volume {ihmt} is not 4D')
    print('ihMT exists')

    # check B1 map
    if b1 is None:
        print(
            'No B1 map provided (this is highly not recommended). '
            'Will use 1 for all voxels.'
        )
    else:
        print('B1 map exists')

    # check mask
    if mask is None:
        print('No mask provided. Will use all voxels.')
    else:
        print('Mask exists')

    if mtdsatb1sq_file is not None and b1 is None:
        raise ValueError('B1 map is required for MTdsatB1sq output.')

    if mtssatb1sq_file is not None and b1 is None:
        raise ValueError('B1 map is required for MTssatB1sq output.')

    if ihmtsatb1sq_file is not None and b1 is None:
        raise ValueError('B1 map is required for ihMTsatB1sq output.')

    #### load data
    # get ihMT data
    ref_img = nb.load(ihmt)
    ihmt_data = ref_img.get_fdata()

    # get B1 data
    if b1 is not None:
        b1_data = nb.load(b1).get_fdata()
    else:
        b1_data = np.ones(ihmt_data.shape[0:3])

    # get T1 data
    t1_data = nb.load(t1).get_fdata()

    # get indices to process from mask
    if mask is not None:
        mask_data = nb.load(mask).get_fdata()
    else:
        mask_data = np.ones(ihmt_data.shape[:3])

    mask_idx = np.asarray(np.where(mask_data == 1))

    #### Sorting and/or exclude with indices -R/-S/-D
    # Default R=1; S=2,4,...N-1; D=3,5,...,N
    if single_offset_idx is None:
        single_offset_idx = np.arange(1, ihmt_data.shape[3], 2)

    if dual_offset_idx is None:
        dual_offset_idx = np.arange(2, ihmt_data.shape[3], 2)

    #### build common xData elements
    t1_data_masked = t1_data[mask_idx[0], mask_idx[1], mask_idx[2]][:, None]
    b1_data_masked = b1_data[mask_idx[0], mask_idx[1], mask_idx[2]][:, None]

    cosFA_RO = np.cos(tflparx['FA'] * b1_data_masked * np.pi / 180)
    Np = np.full((t1_data_masked.shape[0], 1), ihmtparx['Np'])
    Npart = np.full((t1_data_masked.shape[0], 1), tflparx['Npart'])
    E1_subTR = np.exp(
        np.divide(
            -tflparx['subTR'],
            t1_data_masked,
            out=np.ones(t1_data_masked.shape, dtype=float),
            where=t1_data_masked != 0,
        )
    )

    ################ MT0 estimation
    #### build xData_MT0
    if exp_rage_gre == 'RAGE':
        E1_ihMT = np.exp(
            np.divide(
                -(ihmtparx['BTR'] * (ihmtparx['NBTR'] - 1) + ihmtparx['BTRlast']),
                t1_data_masked,
                out=np.ones(t1_data_masked.shape, dtype=float),
                where=t1_data_masked != 0,
            )
        )
        E1_RD = np.exp(
            np.divide(
                -(
                    tflparx['TR']
                    - (ihmtparx['NBTR'] - 1) * ihmtparx['BTR']
                    - ihmtparx['BTRlast']
                    - tflparx['Npart'] * tflparx['subTR']
                ),
                t1_data_masked,
                out=np.ones(t1_data_masked.shape, dtype=float),
                where=t1_data_masked != 0,
            )
        )
    else:
        E1_ihMT = np.exp(
            np.divide(
                -(ihmtparx['Np'] * ihmtparx['dt'] + ihmtparx['pihMTdelay']),
                t1_data_masked,
                out=np.ones(t1_data_masked.shape, dtype=float),
                where=t1_data_masked != 0,
            )
        )
        E1_RD = np.exp(
            np.divide(
                -(
                    tflparx['TR']
                    - ihmtparx['Np'] * ihmtparx['dt']
                    - ihmtparx['pihMTdelay']
                    - tflparx['Npart'] * tflparx['subTR']
                ),
                t1_data_masked,
                out=np.ones(t1_data_masked.shape, dtype=float),
                where=t1_data_masked != 0,
            )
        )

    xData_MT0 = [*zip(np.hstack((E1_ihMT, E1_subTR, E1_RD, cosFA_RO, Npart)), strict=False)]

    #### run pre-compute MT0 once & for all
    print('')
    print('--------------------------------------------------')
    print('-------- Proceeding to MT0 pre-computation -------')
    print('--------------------------------------------------')
    print('')

    start_time = time.time()
    with multiprocessing.Pool(nworkers) as pool:
        RES = pool.starmap(compute_mt0, xData_MT0)
    delay = time.time()
    print(f'---- Done in {delay - start_time} seconds ----')
    Mz_MT0 = np.array([a_tup[0] for a_tup in RES], dtype=float)[:, None]
    A_RAGE = np.array([a_tup[1] for a_tup in RES], dtype=float)[:, None]
    B_RAGE = np.array([a_tup[2] for a_tup in RES], dtype=float)[:, None]

    ################ MTsat estimation
    #### build xData_MTw
    E1_dt = np.exp(
        np.divide(
            -ihmtparx['dt'],
            t1_data_masked,
            out=np.ones(t1_data_masked.shape, dtype=float),
            where=t1_data_masked != 0,
        )
    )
    if exp_rage_gre == 'RAGE':
        NBTR = np.full((t1_data_masked.shape[0], 1), ihmtparx['NBTR'])
        E1_BTR = np.exp(
            np.divide(
                -(ihmtparx['BTR'] - ihmtparx['Np'] * ihmtparx['dt']),
                t1_data_masked,
                out=np.ones(t1_data_masked.shape, dtype=float),
                where=t1_data_masked != 0,
            )
        )
        E1_BTRlast = np.exp(
            np.divide(
                -(ihmtparx['BTRlast'] - ihmtparx['Np'] * ihmtparx['dt']),
                t1_data_masked,
                out=np.ones(t1_data_masked.shape, dtype=float),
                where=t1_data_masked != 0,
            )
        )

        xData = np.hstack((Mz_MT0, A_RAGE, B_RAGE, E1_dt, E1_BTR, E1_BTRlast, Np, NBTR))
    else:
        E1_pihMTdelay = np.exp(
            np.divide(
                -ihmtparx['pihMTdelay'],
                t1_data_masked,
                out=np.ones(t1_data_masked.shape, dtype=float),
                where=t1_data_masked != 0,
            )
        )

        xData = np.hstack((Mz_MT0, A_RAGE, B_RAGE, E1_dt, E1_pihMTdelay, Np))

    #### build yData
    MT0_data = np.mean(ihmt_data[:, :, :, reference_idx], axis=3)
    MTs_data = np.mean(ihmt_data[:, :, :, single_offset_idx], axis=3)
    MTd_data = np.mean(ihmt_data[:, :, :, dual_offset_idx], axis=3)

    MT0_ydata = MT0_data[mask_idx[0], mask_idx[1], mask_idx[2]][:, None]
    MTs_yData = MTs_data[mask_idx[0], mask_idx[1], mask_idx[2]][:, None] / MT0_ydata
    MTd_yData = MTd_data[mask_idx[0], mask_idx[1], mask_idx[2]][:, None] / MT0_ydata

    #### iterable lists
    exp_rage_gre = [exp_rage_gre] * xData.shape[0]
    xtol = [xtol] * xData.shape[0]
    MTs_iterable = [*zip(xData, MTs_yData, exp_rage_gre, xtol, strict=False)]
    MTd_iterable = [*zip(xData, MTd_yData, exp_rage_gre, xtol, strict=False)]

    #### run
    print('')
    print('--------------------------------------------------')
    print('--------- Proceeding to MTssat estimation --------')
    print('--------------------------------------------------')
    print('')

    start_time = time.time()
    with multiprocessing.Pool(nworkers) as pool:
        MTssat = pool.starmap(fit_mtsat_brentq, MTs_iterable)
    delay = time.time()
    print(f'---- Done in {delay - start_time} seconds ----')
    MTssat = np.array(MTssat, dtype=float)

    print('')
    print('--------------------------------------------------')
    print('--------- Proceeding to MTdsat estimation --------')
    print('--------------------------------------------------')
    print('')

    start_time = time.time()
    with multiprocessing.Pool(nworkers) as pool:
        MTdsat = pool.starmap(fit_mtsat_brentq, MTd_iterable)
    delay = time.time()
    print(f'---- Done in {delay - start_time} seconds ----')
    MTdsat = np.array(MTdsat, dtype=float)

    ################ store & save NIfTI(single_offset_idx)
    ihMTsat_map = np.full(ref_img.shape[0:3], 0, dtype=float)
    ihMTsat_map[mask_idx[0], mask_idx[1], mask_idx[2]] = 2 * (MTdsat - MTssat) * 100
    ihMTsat_map[(ihMTsat_map < 0) | (ihMTsat_map > 100)] = 0
    new_img = nb.Nifti1Image(ihMTsat_map, ref_img.affine, ref_img.header)
    nb.save(new_img, ihmtsat_file)

    if mtdsat_file is not None:
        MTdsat_map = np.full(ref_img.shape[0:3], 0, dtype=float)
        MTdsat_map[mask_idx[0], mask_idx[1], mask_idx[2]] = MTdsat * 100
        new_img = nb.Nifti1Image(MTdsat_map, ref_img.affine, ref_img.header)
        nb.save(new_img, mtdsat_file)

    if mtssat_file is not None:
        MTssat_map = np.full(ref_img.shape[0:3], 0, dtype=float)
        MTssat_map[mask_idx[0], mask_idx[1], mask_idx[2]] = MTdsat * 100
        new_img = nb.Nifti1Image(MTssat_map, ref_img.affine, ref_img.header)
        nb.save(new_img, mtssat_file)

    if ihmtsatb1sq_file is not None and b1 is not None:
        ihMTsatB1sq_map = np.full(ref_img.shape[0:3], 0, dtype=float)
        ihMTsatB1sq_map = np.divide(
            ihMTsat_map,
            b1_data**2,
            out=np.zeros(ihMTsat_map.shape, dtype=float),
            where=b1_data != 0,
        )
        new_img = nb.Nifti1Image(ihMTsatB1sq_map, ref_img.affine, ref_img.header)
        nb.save(new_img, ihmtsatb1sq_file)

    if mtdsatb1sq_file is not None and b1 is not None:
        MTdsatB1sq_map = np.full(ref_img.shape[0:3], 0, dtype=float)
        MTdsatB1sq_map[mask_idx[0], mask_idx[1], mask_idx[2]] = MTdsat * 100
        MTdsatB1sq_map = np.divide(
            MTdsat_map,
            b1_data**2,
            out=np.zeros(MTdsat_map.shape, dtype=float),
            where=b1_data != 0,
        )
        new_img = nb.Nifti1Image(MTdsatB1sq_map, ref_img.affine, ref_img.header)
        nb.save(new_img, mtdsatb1sq_file)

    if mtssatb1sq_file is not None and b1 is not None:
        MTssatB1sq_map = np.full(ref_img.shape[0:3], 0, dtype=float)
        MTssatB1sq_map[mask_idx[0], mask_idx[1], mask_idx[2]] = MTssat * 100
        MTssatB1sq_map = np.divide(
            MTssat_map,
            b1_data**2,
            out=np.zeros(MTssat_map.shape, dtype=float),
            where=b1_data != 0,
        )
        new_img = nb.Nifti1Image(MTssatB1sq_map, ref_img.affine, ref_img.header)
        nb.save(new_img, mtssatb1sq_file)


def compute_ab(TILT, E1_SHORT, E1_LONG, N):
    ## compute closed forms of Mz+1=A*Mz+B for a N*(TILT+E1_SHORT)+E1_LONG sequence pattern
    A = E1_LONG * (E1_SHORT * TILT) ** N
    B = (
        1
        + E1_LONG * (E1_SHORT * TILT - (E1_SHORT * TILT) ** N) / (1 - E1_SHORT * TILT)
        - E1_LONG / TILT * (E1_SHORT * TILT * (1 - (E1_SHORT * TILT) ** N) / (1 - E1_SHORT * TILT))
    )
    return A, B


def compute_mt0(xData):
    E1_ihMT, E1_subTR, E1_RD, cosFA_RO, Npart = xData
    A_RAGE, B_RAGE = compute_ab(cosFA_RO, E1_subTR, E1_RD, Npart)
    A_ihMT, B_ihMT = E1_ihMT, 1 - E1_ihMT

    Mz_MT0 = (B_ihMT + A_ihMT * B_RAGE) / (1 - A_ihMT * A_RAGE)

    return Mz_MT0, A_RAGE, B_RAGE


def compute_mtsat_rage_root(delta, xData, yData):
    Mz_MT0, A_RAGE, B_RAGE, E1_dt, E1_BTR, E1_BTRlast, Np, NBTR = xData
    A_BTR, B_BTR = compute_ab(1 - delta, E1_dt, E1_BTR, Np)
    A_BTRL, B_BTRL = compute_ab(1 - delta, E1_dt, E1_BTRlast, Np)

    Mz_MTw = (
        B_BTRL
        + B_BTR * A_BTRL * (A_BTR - A_BTR**NBTR) / (A_BTR - A_BTR**2)
        + B_RAGE * A_BTRL * A_BTR ** (NBTR - 1)
    ) / (1 - A_BTRL * A_RAGE * A_BTR ** (NBTR - 1))

    return Mz_MTw / Mz_MT0 - yData


def compute_mtsat_gre_root(delta, xData, yData):
    Mz_MT0, A_RAGE, B_RAGE, E1_dt, E1_pihMTdelay, Np = xData
    A_BTR, B_BTR = compute_ab(1 - delta, E1_dt, E1_pihMTdelay, Np)

    Mz_MTw = (B_BTR + A_BTR * B_RAGE) / (1 - A_BTR * A_RAGE)

    return Mz_MTw / Mz_MT0 - yData


def fit_mtsat_brentq(xData, yData, exp_rage_gre, xtol):
    # root-finding with Brent'single_offset_idx method
    # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html#scipy.optimize.brentq

    try:
        if exp_rage_gre == 'RAGE':
            x0 = scipy.optimize.brentq(compute_mtsat_rage_root, 0, 0.3, (xData, yData), xtol=xtol)
        else:
            x0 = scipy.optimize.brentq(compute_mtsat_gre_root, 0, 0.3, (xData, yData), xtol=xtol)
        return x0
    except:
        return 0


def get_cpu_count():
    # from joblib source code (commit d5c8274)
    # https://github.com/joblib/joblib/blob/master/joblib/externals/loky/backend/context.py#L220-L246
    if sys.platform == 'linux':
        cpu_info = subprocess.run('lscpu --parse=core'.split(' '), capture_output=True)
        cpu_info = cpu_info.stdout.decode('utf-8').splitlines()
        cpu_info = {line for line in cpu_info if not line.startswith('#')}
        cpu_count_physical = len(cpu_info)
    elif sys.platform == 'win32':
        cpu_info = subprocess.run(
            'wmic CPU Get NumberOfCores /Format:csv'.split(' '), capture_output=True
        )
        cpu_info = cpu_info.stdout.decode('utf-8').splitlines()
        cpu_info = [l.split(',')[1] for l in cpu_info if (l and l != 'Node,NumberOfCores')]
        cpu_count_physical = sum(map(int, cpu_info))
    elif sys.platform == 'darwin':
        cpu_info = subprocess.run('sysctl -n hw.physicalcpu'.split(' '), capture_output=True)
        cpu_info = cpu_info.stdout.decode('utf-8')
        cpu_count_physical = int(cpu_info)
    else:
        raise NotImplementedError(f'unsupported platform: {sys.platform}')
    if cpu_count_physical < 1:
        raise ValueError(f'found {cpu_count_physical} physical cores < 1')
    return cpu_count_physical


def is_valid_file(parser, arg):
    """Check if argument is existing file."""
    if not os.path.isfile(arg) and arg is not None:
        parser.error(f'The file {arg} does not exist!')

    return arg


def _main(argv=None):
    """Main function for fit_ihMTsat."""
    parser = _get_parser()
    args = parser.parse_args(argv)
    main(**vars(args))


if __name__ == '__main__':
    sys.exit(_main())
