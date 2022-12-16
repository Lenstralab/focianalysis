import os
import shutil
import sys
import re
import random
import numpy as np
import pandas as pd
from numbers import Number
from tqdm.auto import tqdm
from tiffwrite import tiffwrite
from time import time
from tllab_common.wimread import imread as imr
from tllab_common import misc
from tllab_common.findcells import findcells
from smfish import FISH_pipeline as fish
from datetime import datetime
from sklearn import neighbors
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

if __package__ is None or __package__ == '':
    import _version
else:
    from . import _version


def setup_dirs(params):
    """
    Setup directories
    Create list of files but removes hidden files starting with "."
    """
    file_list_in = [f for f in os.listdir(params['path_in']) if not f.startswith(".")]
    file_list_in.sort()
    if "beads.czi" in file_list_in:
        file_list_in.remove("beads.czi")
    file_list_in = [x for x in file_list_in if x[-4:] == ".czi"]
    if params['processAll'] == 0:
        file_list_in = file_list_in[params['processFrom']:params['processTo']]

    # create file list and output directory
    date = params['path_in'].split("/")[-2]
    pathOut = params['path_out'] + date + "/"
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)
    date = date[:8]  # shorten date length for filenames (format = yyyymmdd)

    paramsexp = {'pathOut': pathOut, 'date': date, 'zSlices': None if params['zSlices'] is None else
                 range(params['zSlices']) if isinstance(params['zSlices'], Number) else range(*params['zSlices'])}
    return paramsexp, file_list_in


def make_filenames(lfn, paramsexp):
    """ Create textfile with all image indexing parameters """
    print("Writing textfile containing image names, indexes and replicate combinations")

    # create names_idx (= image_to_combine) list of image indexes-list to combine per strain + condition name
    names_img = np.array([name.rsplit('_', 1)[0] for name in lfn])  # Names of strain + condition for each image
    names_unique = np.unique(names_img).tolist()  # Unique names of strain + condition names for all images
    names_idx = np.array([np.where(names_img == name)[0].tolist() for name in np.unique(names_img)], dtype=object).tolist()

    # create com_idx (= combine_names) indicating the name in names_unique belonging to the images in imageToCombine
    com_strain = np.array([re.search('YTL\d+', name).group(0) + ' ' for name in names_unique])
    com_media = np.array([name.rsplit('_', 1)[1] for name in names_unique])
    com_condition = np.char.add(com_strain, com_media)
    com_idx = np.array([np.where(name == np.unique(com_condition))[0][0] for name in com_condition]).tolist()

    # sort com_idx and names_idx simultaneously
    com_idx, names_idx,  = [list(elem) for elem in zip(*sorted(zip(com_idx, names_idx)))]

    # create string of conditionNames with all unique yeast + sugar names on a new line
    str_con_names = '\nconditionNames: ['
    for name in np.unique(com_condition):
        str_con_names += '\n    \'' + name + '\','
    str_con_names = str_con_names[:-1] + '\n]'

    # create string of imageToCombine with for each unique yeast + sugar name the images to combine on a new line
    str_img_to_combine = '\nimageToCombine: ['
    indexOld = 'N/A'
    for i in range(len(com_idx)):
        if com_idx[i] != indexOld:  # add enter for new unique strain name + condition
            str_img_to_combine += '\n    '
        str_img_to_combine += str(names_idx[i]) + ', '
        indexOld = com_idx[i]
    str_img_to_combine = str_img_to_combine[:-1] + '\n]'

    # create string of combineNames
    str_com_names = '\ncombineNames: ' + str(com_idx)

    # create string of None values for maskThreshold = [None]
    str_mask_thr = '\nmaskThreshold: [\n'
    none_lst = len(lfn)//10 * ('    ' + 10 * 'None, ' + '\n') + '    ' + (len(lfn) % 10) * 'None, '
    str_mask_thr += none_lst + '\n]'

    # create text string containing all text and save this in a textfile
    text = [str(i) + ' ' + lfn[i] for i in range(len(lfn))]  # annotates indexes in image_to_combine
    text.append(str_con_names)
    text.append(str_img_to_combine)
    text.append(str_com_names)
    text.append(str_mask_thr)
    np.savetxt(f"{paramsexp['pathOut']}{paramsexp['date']}_all_filenames_index.txt", text, "%s", "\t")


def make_masks(lfn, params, paramsexp):
    """
    Make cellular and nuclear masks from max projections
    Uses 2nd chanel for two-color data
    """
    print('Making masks')
    for i, fn in enumerate(lfn):
        file = os.path.splitext(fn)[0]
        path = os.path.join(paramsexp['pathOut'], paramsexp['date'])
        if not os.path.exists(f'{path}_{file}_max_nucleus_mask.tif'):
            with imr(f'{path}_{os.path.splitext(fn)[0]}_max.tif', dtype=float) as max_im:
                if len(params['channelsToAnalyze']) == 2:
                    img_file = max_im(1)  # use 2nd channel for two-color data
                else:
                    img_file = max_im(0)

            if params.get('cornerMask', False):
                corner_mask = np.where(img_file < params.get('cornerThreshold', 0))
                img_file[corner_mask] = np.nan

            cells, nuclei = findcells(img_file, ccdist=40, threshold=params['maskThreshold'][i])
            tiffwrite(f'{path}_{file}_max_nucleus_mask.tif', nuclei.astype('uint16'), colormap='glasbey',
                      pxsize=max_im.pxsize, deltaz=max_im.deltaz)
            tiffwrite(f'{path}_{file}_max_cell_mask.tif', cells.astype('uint16'), colormap='glasbey',
                      pxsize=max_im.pxsize, deltaz=max_im.deltaz)


def check_masks(lfn, paramsexp):
    """
    Check whether nuclear and cellular masks match up after manual checking/removing/adjusting cell masks.
    """
    print("Checking masks")
    for fn in lfn:
        file = os.path.splitext(fn)[0]
        path = os.path.join(paramsexp['pathOut'], paramsexp['date'])

        # if old mask file names are used, rename files
        if not os.path.isfile(f"{path}_{file}_max_nucleus_mask.tif"):
            os.rename(f"{path}_{file}_nucleus_mask.tif", f"{path}_{file}_max_nucleus_mask.tif")
        if not os.path.isfile(f"{path}_{file}_max_cell_mask.tif"):
            os.rename(f"{path}_{file}_cells_mask.tif", f"{path}_{file}_max_cell_mask.tif")

        # open mask files
        with imr(f"{path}_{file}_max_nucleus_mask.tif", dtype=int) as im:
            nuc_mask = im(0)
        with imr(f"{path}_{file}_max_cell_mask.tif", dtype=int) as im:
            cell_mask = im(0)

        # warn for inconsistencies between cell and nucleus masks
        c, n = set(cell_mask.flatten()), set(nuc_mask.flatten())
        if not c == n:
            if c - n:
                print(f'Cell(s) {c - n} in {path}_{file} do(es) not have a nucleus.')
            if n - c:
                print(f'Nucle(us/i) {n - c} in {path}_{file} do(es) not have a cell.')

    print("Masks are checked")


def get_z_correction(lfn, params, paramsexp):
    """
    Localize beads in two-color beads.czi file and calculate dz correction in zSlice size
    """
    print("Calculating z correction")

    # step 1: localize beads
    bead_params = {'localize': params['localize'].copy(), 'channelsToAnalyze': [0, 1]}
    bead_params['localize']['threshold'] = {'method': 'SD', 'ch1': 10, 'ch2': 10}
    fish.run_localize(params['path_in'], ('beads.czi',), bead_params, paramsexp, ('ch1', 'ch2'))

    # step 2: calculate z correction
    path_loc = os.path.join(paramsexp['pathOut'], paramsexp['date'])
    c = find_dz(f"{path_loc}_beads_loc_results_ch1.txt", 0.09707, 0.250)
    c.to_csv(f"{path_loc}_beads_z_correction.txt", sep='\t')

    print(f"dz = {np.mean(c['dz']):.3f}±{np.std(c['dz']) / np.sqrt(len(c)):.3f} px")


def find_nn(r, g, pxsize, deltaz, lsuffix='_ch1', rsuffix='_ch2'):
    nan = 1e99
    d = cdist(r[['x', 'y', 'z']] * (pxsize, pxsize, deltaz), g[['x', 'y', 'z']] * (pxsize, pxsize, deltaz))
    s = np.max(d.shape)
    d = np.pad(d, ((0, s - d.shape[0]), (0, s - d.shape[1])), constant_values=nan)
    n = neighbors.NearestNeighbors(n_neighbors=1, radius=10, metric='precomputed')
    n.fit(d)
    distances, indices = [i.T[0].tolist() for i in n.kneighbors(d)]

    def check(i, d):
        return not (d == nan or i == 0 or
                    (indices.count(i) > 1 and d > min([d for d, j in zip(distances, indices) if j == i])))
    indices, distances = zip(*[id if check(*id) else (None, None) for id in zip(indices, distances)])
    c = r.assign(gi=indices[:len(r)], d=distances[:len(r)]) \
         .dropna(subset='gi') \
         .set_index('gi') \
         .join(g.reset_index(drop=True), lsuffix=lsuffix, rsuffix=rsuffix) \
         .reset_index(drop=True)
    # remove outliers using the interquartile range rule
    return c.query(f"d<{4*c['d'].quantile(0.75) - 3*c['d'].quantile(0.25)}")


def find_dz(r_file, pxsize, deltaz):
    r = pd.read_table(r_file).query('fit==True')  # red channel
    g = pd.read_table(r_file.replace('results_ch1.txt', 'results_ch2.txt')).query('fit==True')  # green channel
    c = find_nn(r, g, pxsize, deltaz)
    c = c.assign(dz=c['z_ch2'] - c['z_ch1'], ddz=np.sqrt(c['dz_ch2'] ** 2 + c['dz_ch1'] ** 2))

    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.plot(r['x'], r['y'], 'or', mfc='none')
    plt.plot(g['x'], g['y'], 'xg', mfc='none')
    plt.title('all fits')

    plt.subplot(2, 2, 2)
    for _, row in c.iterrows():
        plt.plot(row[['x_ch1', 'x_ch2']], row[['y_ch1', 'y_ch2']], 'o-')
    plt.title('all pairs')

    plt.subplot(2, 2, 3)
    plt.hist(c['dz'], 15)
    plt.xlabel('dz (deltaz)')
    plt.ylabel('#')
    plt.title(f"dz = {np.mean(c['dz']):.3f}±{np.std(c['dz']) / np.sqrt(len(c)):.3f} px")

    plt.subplot(2, 2, 4)
    plt.hist(c['ddz'], 15)
    plt.xlabel('error in dz (deltaz)')
    plt.ylabel('#')

    plt.savefig(f"{r_file[:r_file.find('loc_results')]}z_correction.pdf", bbox_inches='tight')
    plt.close()

    return c


def combine_foci(lfn, params, paramsexp):
    """
    Combine foci data (per channel) per replicate. Stitches up localize files.
    """
    print("Combining foci data per replicate")
    path = os.path.join(paramsexp['pathOut'], paramsexp['date'])

    # create list with names of replicates
    rep_names = []
    rep_nrs = []
    rep_nr = 0

    for i in params['combineNames']:
        if i in rep_nrs:
            rep_nr += 1
        else:
            rep_nr = 0
        rep_nrs.append(i)
        rep_names.append(f"{params['conditionNames'][i]}_{rep_nr}")

    # combine localize files per replicate and add empty cells so that replicate loc_results contains all cells
    for channel in params['channelsToAnalyze']:
        color = f'ch{channel + 1}'
        for i in range(len(params['imageToCombine'])):
            df_loc_results_rep = pd.DataFrame()
            for j in params['imageToCombine'][i]:
                fn = os.path.splitext(lfn[j])[0]
                # load loc results of image
                df_loc_results_img = pd.read_csv(f"{path}_{fn}_loc_results_{color}_{params['localize']['threshold']['method']}.txt", sep='\t')

                # make dataframe of all masks from image. As masks are checked before, unique cell_masks_img == nuc_masks_img
                with imr(f"{path}_{fn}_max_cell_mask.tif", dtype=int) as im:
                    cell_masks_img = nuc_masks_img = np.delete(np.unique(im(0)), 0)
                df_masks_img = pd.DataFrame(list(zip(cell_masks_img, nuc_masks_img)), columns=['cell', 'nucleus'])

                # join df_loc_results_img and df_masks_img; empty cells have NaN for localize parameters
                df_loc_results_img = df_masks_img.set_index(['cell', 'nucleus']).join(df_loc_results_img.set_index(['cell', 'nucleus'])).reset_index()

                # add filename as column and add to loc results of replicate
                df_loc_results_img['file'] = fn
                df_loc_results_rep = pd.concat((df_loc_results_rep, df_loc_results_img), ignore_index=True)

            # correct for difference in z for second channel
            if color == 'ch2' and os.path.exists(f"{path}_beads_z_correction.txt"):
                dz = np.mean(pd.read_csv(f"{path}_beads_z_correction.txt", sep='\t', usecols=['dz'])['dz'])
                df_loc_results_rep['z'] = df_loc_results_rep['z'] + dz
                print(f"z corrected with dz = {dz:.3f} px")

            df_loc_results_rep = df_loc_results_rep.set_index(['file', 'cell', 'nucleus'])
            df_loc_results_rep.to_csv(f"{path}_loc_results_{color}_{rep_names[i].replace(' ', '_')}.txt", sep='\t', index=True)


def get_random_position(lfn, params, paramsexp):
    """
    Pick a random position (in pixels) for each nuclear mask
    """
    print("Getting random position within each nucleus")
    path = os.path.join(paramsexp['pathOut'], paramsexp['date'])

    # create list with names of replicates
    rep_names = []
    rep_nrs = []
    rep_nr = 0

    for i in params['combineNames']:
        if i in rep_nrs:
            rep_nr += 1
        else:
            rep_nr = 0
        rep_nrs.append(i)
        rep_names.append(f"{params['conditionNames'][i]}_{rep_nr}")

    # make table with for each nuclear mask a random (RDM) position and the Centre Of Mass (COM) per replicate
    for i in range(len(params['imageToCombine'])):
        df_random_rep = pd.DataFrame(columns=['file', 'cell', 'y', 'x'])
        for j in params['imageToCombine'][i]:
            fn = os.path.splitext(lfn[j])[0]
            with imr(f"{path}_{fn}_max_nucleus_mask.tif", dtype=int) as im:
                nuc_mask_array = im(0)
                nuc_list = np.delete(np.unique(nuc_mask_array), 0)
                for nuc in nuc_list:
                    nuc_loc = np.where(nuc_mask_array == nuc)  # Matrix with y and x coordinates of nuclear mask
                    nuc_mask = nuc_mask_array * 0  # Matrix of size nucArray in which each pixel has a value of 0
                    nuc_mask[nuc_loc] = 1  # Change pixel values from 0 to 1 for all nuclear mask pixels
                    # COM_x, COM_y = ndimage.measurements.center_of_mass(nuc_mask)  # Get x and y coordinates of center of mass
                    nuc_RDM_index = random.randrange(0, nuc_loc[0].shape[0], 1)  # Pick random nuclear mask pixel index
                    y_RDM, x_RDM = [nuc_loc[0][nuc_RDM_index], nuc_loc[1][nuc_RDM_index]]  # Get the y and x coordinates of the random pixel
                    df_random_rep.loc[df_random_rep.shape[0]] = [fn, nuc, y_RDM, x_RDM]
        df_random_rep.to_csv(f"{path}_loc_random_{rep_names[i].replace(' ', '_')}.txt", sep='\t', index=False)


# TODO: ask if this is still necessary
def save_optimize_threshold_data(lfn, params, paramsexp):
    """ Save all data of RunOptimizeThresh """
    for channel in params['channelsToAnalyze']:
        color = f'ch{channel + 1}'
        path = os.path.join(paramsexp['pathOut'], paramsexp['date'])

        if params['localize']['threshold']['method'].upper() == 'SD':
            tbins = 1
            thresholds = np.array(np.arange(1, 51, tbins))
        else:
            tbins = 25
            thresholds = np.array(range(100, 1350, tbins))

        for fn in lfn:
            d = np.loadtxt(f"{path}_{os.path.splitext(fn)[0]}_threshold_"
                           f"optimization_{color}_{params['localize']['threshold']['method']}.txt")
            thresholds = np.vstack((thresholds, d))

        np.savetxt(f"{path}_all_thresholds_{color}_{params['localize']['threshold']['method']}.txt", thresholds,
                   "%.8e", "\t")
        np.savetxt(f"{path}_all_filenames_{color}_{params['localize']['threshold']['method']}.txt", lfn, "%s", "\t")


def foci_pipeline(parameter_file):
    params = misc.getParams(parameter_file, __file__.replace('.py', '_parameters_template.yml'),
                            ('path_in', 'path_out'))
    params['focianalysis'] = True
    paramsexp, lfn = setup_dirs(params)

    # Save parameter file and pipeline version
    date_time = datetime.now()
    date_time = date_time.strftime("%Y-%m-%d_%Hh%Mm%Ss")
    shutil.copyfile(parameter_file,
                    os.path.join(paramsexp['pathOut'], f"{paramsexp['date']}_runtime_{date_time}_{os.path.basename(parameter_file)}"))
    shutil.copyfile(os.path.abspath(_version.__file__),
                    os.path.join(paramsexp['pathOut'], f"{paramsexp['date']}_runtime_{date_time}_foci_pipeline_version.txt"))

    if params.get('MakeFilenames', False):
        make_filenames(lfn, paramsexp)

    if params.get('MaxProj', False) or params.get('MakeMaxProjection', False):
        fish.max_projection(params['path_in'], lfn, paramsexp)

    if params.get('MakeMasks', False):
        make_masks(lfn, params, paramsexp)

    if params.get('CheckMasks', False):
        check_masks(lfn, paramsexp)

    if params.get('RunOptimizeThresh', False):
        for fn in tqdm(lfn, desc="Optimizing Threshold"):
            fish.run_optimize_threshold(params['path_in'], [fn], params, paramsexp, ('ch1', 'ch2'))

    if params.get('RunLocalize', False):
        fish.run_localize(params['path_in'], lfn, params, paramsexp, ('ch1', 'ch2'))

    if params.get('MakeMaskImage', False):
        fish.make_mask_image(lfn, params, paramsexp, ('ch1', 'ch2'))

    if params.get('RunOptimizeThresh', False):
        save_optimize_threshold_data(lfn, params, paramsexp)

    if params.get('zCorrection', False):
        get_z_correction(lfn, params, paramsexp)

    if params.get('CombineFoci', False):
        combine_foci(lfn, params, paramsexp)

    if params.get('RDMposition', False):
        get_random_position(lfn, params, paramsexp)


def main():
    misc.ipy_debug()
    if len(sys.argv) < 2:
        if os.path.exists('foci_pipeline_parameters.yml'):
            parameter_file = 'foci_pipeline_parameters.yml'
        else:
            raise FileNotFoundError('Could not find the parameter file.')
        print('Using ' + parameter_file)
    else:
        parameter_file = sys.argv[1]

    tm = time()
    foci_pipeline(parameter_file)

    print('------------------------------------------------')
    print(misc.color('Pipeline finished, took {} seconds.'.format(time()-tm), 'g:b'))


if __name__ == '__main__':
    main()
    imr.kill_vm()
