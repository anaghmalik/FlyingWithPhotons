import numpy as np
from dataset_utils import read_h5
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from joblib import Parallel, delayed
import tqdm
import os 
import imageio
import h5py
    

def generate_list(input_list):
    result = []
    for index, count in enumerate(input_list):
        result.extend([index] * count)
    return result

def apply_in_parallel_ncc(data, pulse, n_jobs=45):
    lm, direct, gmm, gaussian_pixels = zip(*Parallel(n_jobs=n_jobs)(delayed(single_pixel_solve)(element, pulse) for element in tqdm.tqdm(data, total=len(data))))
    return lm, direct, gmm, gaussian_pixels

def get_pulse(pulse):
    # center = np.argmax(pulse)
    center = np.arange(pulse.shape[0])*pulse
    center = int(center.sum()/pulse.sum())
    pulse = pulse[center-50:center+51]
     
    pulse = pulse
    pulse = pulse/pulse.sum()
    return pulse

def single_pixel_solve(transient, normalized_pulse, pixel=(400, 200), num_gaussians=5, direct_dist_thold=30):
    pixel = transient.squeeze()
    samples = pixel.astype(np.uint32)
    data = np.array(generate_list(samples))
    if data.shape[0]<6:
        return np.zeros((pixel.shape[-1])), np.zeros((pixel.shape[-1])), None, np.zeros((pixel.shape[-1]))
    gmm = GaussianMixture(n_components=num_gaussians) 
    gmm.fit(data.reshape(-1, 1))  
    
    t = np.arange(0, pixel.shape[-1])
    mins = np.array([900., 900., 900.])
    gaussian_pixels = 0 

    for weight, mean, covar in zip(gmm.weights_, gmm.means_, gmm.covariances_):
        eval = weight*norm.pdf(t, mean, np.sqrt(covar)).ravel()
        gaussian_pixels += eval
        if mean<mins[0] and weight>0:
            mins[0], mins[1], mins[2] = mean, covar, weight
            
    gaussian_pixels = gaussian_pixels.astype(np.float32)
    normalized_transient = (gaussian_pixels - np.mean(gaussian_pixels))
    normalized_transient /= np.linalg.norm(normalized_transient)
    
    lm = np.correlate(normalized_transient, normalized_pulse, 'same')
    
    direct_pos = np.argmax(lm, -1)
    selector = np.logical_or((np.abs(gmm.means_ - direct_pos)<direct_dist_thold).squeeze(), gmm.means_.squeeze() < direct_pos.squeeze())
    
    direct = np.zeros((pixel.shape[-1]))
    for weight, mean, covar in zip(gmm.weights_[selector], gmm.means_[selector], gmm.covariances_[selector]):
        eval = weight*norm.pdf(t, mean, np.sqrt(covar)).ravel()
        direct += eval
    
    direct *= pixel.sum()/gaussian_pixels.sum()
    gaussian_pixels *= pixel.sum()/gaussian_pixels.sum()

    return lm, direct, gmm, gaussian_pixels
    
def get_video_fram_transient(transient, path):
    frames = np.stack([transient, transient, transient], axis=-1)
    images = []
    mx = transient.max()
    writer = imageio.get_writer(path, fps=20)


    for i in range(0, transient.shape[-1]):
        img = (255*(frames[..., i, :]/mx)**(1/2.2)).astype(np.uint8)
        writer.append_data(img)
    writer.close()

def create_multiview_dataset(basedir, outpath, pulse_path, ran=(1500, -1500), output_shape=3000, ncc_thold = 0.5):
    files = sorted(os.listdir(basedir))
    files = [f for f in files if "h5" in f]
    files = [f for f in files if f[0].isdigit()][::-1]
    pulse = np.load(pulse_path)
    normalized_pulse = (pulse - np.mean(pulse))
    normalized_pulse /= np.linalg.norm(pulse)
    
    os.makedirs(os.path.join(outpath, "direct"), exist_ok=True)
    os.makedirs(os.path.join(outpath, "indirect"), exist_ok=True)
    os.makedirs(os.path.join(outpath, "images"), exist_ok=True)
    
    for file in files:
        file_base = os.path.splitext(os.path.basename(file))[0]
        transient = read_h5(os.path.join(basedir, file)).squeeze()[..., ran[0]:ran[1]]
        
        lm, direct, gmm, gaussian_pixels = apply_in_parallel_ncc(transient.reshape(-1, transient.shape[-1]), normalized_pulse)
        
        lm = np.array(lm).reshape(transient.shape[0], transient.shape[1], -1)
        direct = np.array(direct).reshape(transient.shape[0], transient.shape[1], -1)
        gaussian_pixels = np.array(gaussian_pixels).reshape(transient.shape[0], transient.shape[1], -1)

        direct[lm.max(-1)<ncc_thold] = 0
        indirect = np.clip(gaussian_pixels-direct, 0, None)

        plt.subplot(1, 3, 1)
        plt.title("transient")
        plt.imshow(gaussian_pixels.sum(-1))
        plt.subplot(1, 3, 2)
        plt.title("direct")
        plt.imshow(direct.sum(-1))
        plt.subplot(1, 3, 3)
        plt.title("indirect")
        plt.imshow(indirect.sum(-1))
        plt.tight_layout()
        plt.savefig(f"{outpath}/images/{file_base}.png")
        
        get_video_fram_transient(indirect, f"{outpath}/images/{file_base}_indirect.mp4")
        get_video_fram_transient(direct, f"{outpath}/images/{file_base}_direct.mp4")
        
        direct_big = np.zeros((transient.shape[0], transient.shape[1], output_shape, 1))
        indirect_big = np.zeros((transient.shape[0], transient.shape[1], output_shape, 1))
        direct_big[..., ran[0]:ran[0]+transient.shape[-1], 0]= direct
        indirect_big[..., ran[0]:ran[0]+transient.shape[-1], 0] = indirect
        
        file = h5py.File(f"{outpath}/direct/{file_base}.h5", 'w')
        dataset = file.create_dataset(
        "dataset", direct_big.shape, dtype='f', data=direct_big
        )
        file.close()
        
        file = h5py.File(f"{outpath}/indirect/{file_base}.h5", 'w')
        dataset = file.create_dataset(
        "dataset", indirect_big.shape, dtype='f', data=indirect_big
        )
        file.close()

    

if __name__=="__main__":
    basedir = ""
    outpath = ""
    pulse_path = ""
    create_multiview_dataset(basedir=basedir, outpath=outpath, pulse_path=pulse_path)
    
