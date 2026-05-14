""" example for unfolding a broad 14 MeV peak spectrum 
from reaction rates and response functions with uncertainties. 
options for running in sequential or in parallel
"""
from nfoils.unfold import BayesianUnfolding
import numpy as np
import multiprocessing as mp
import sys
sys.path.append('../')
from process_foil_data import build_response_matrix_from_processed_data

# get response matrix, reaction rates and reaction rate uncertainties from processed data
response_matrix, responses, response_errs = build_response_matrix_from_processed_data(angle=90)

# remove redundant rows or all zero rows from response matrix and corresponding entries in responses and response_errs
# Detect fully-zero rows (within tol) and duplicate rows, remove them and corresponding entries
tol = 1e-30
row_norms = np.linalg.norm(response_matrix, axis=1)
zero_rows = np.where(row_norms <= tol)[0].tolist()

# find duplicate rows (keep first occurrence)
dup_rows = []
for i in range(response_matrix.shape[0]):
    if i in zero_rows or i in dup_rows:
        continue
    for j in range(i + 1, response_matrix.shape[0]):
        if j in zero_rows or j in dup_rows:
            continue
        if np.allclose(response_matrix[i], response_matrix[j], rtol=1e-6, atol=tol):
            dup_rows.append(j)

remove_rows = sorted(set(zero_rows + dup_rows))
if remove_rows:
    print(f"Removing redundant response rows: {remove_rows}")
    keep_mask = np.ones(response_matrix.shape[0], dtype=bool)
    keep_mask[remove_rows] = False
    response_matrix = response_matrix[keep_mask, :]
    responses = responses[keep_mask]
    response_errs = response_errs[keep_mask]


np.savetxt("data/reaction_rates.txt", responses.reshape(-1,1))
np.savetxt("data/reaction_rate_uncertainties.txt", response_errs.reshape(-1,1))
np.savetxt("data/response_matrix.txt", response_matrix, delimiter=",")
response_matrix_uncertainty = 0.05 * response_matrix
np.savetxt("data/response_matrix_uncertainties.txt", response_matrix_uncertainty, delimiter=",")

group_structure = np.append(np.logspace(0, 6, num=15), [3.0e6, 6.0e6, 9.0e6, 12.0e6, 16.0e6])
# The number of flux bins must match the number of response-matrix columns.
if response_matrix.shape[1] != len(group_structure) - 1:
    raise ValueError(
        f"Energy-group mismatch: response_matrix has {response_matrix.shape[1]} columns "
        f"but group_structure defines {len(group_structure) - 1} groups. "
        "Rebuild the response matrix for this group structure, or change group_structure "
        "to have response_matrix.shape[1] + 1 edges."
    )

# write group_structure to file
np.savetxt("data/group_structure.txt",
           group_structure.reshape(1, -1),
           delimiter=",")



# path to files json containing paths for all the unfolding data
# should include response matrix with uncertainties, group structure
# and reaction rates with uncertainties
files_json = 'files.json'

# how many parameters in the model
nparam = 5

# names of parameters
param_names = ['sigma', 'gaussian_peak', 'tail_A', 'tail_alpha', 'tail_E0']

# set initial guesses for each parameter
guesses = [0.1, 1e5, 1e-2, -0.7, 5]  # sigma, peak, tail_A, tail_alpha, tail_E0

# load the unfolding data and parameter info
unfold = BayesianUnfolding(files_json,nparam,param_names,guesses)

# create neutron flux model, given parameters theta 
# and a given neutron energy (MeV)
def model(theta,energy):

    # unpack the tuple of parameters
    # sigma,peak = theta
    sigma, gaussian_peak, tail_A, tail_alpha, tail_E0 = theta

    # set fixed mean energy of the gaussian
    mean_energy = 14.1

    # create the model and return the flux
    # can import generic distributions from the module or make your own
    # flux = unfold.gaussian(mean_energy,sigma,peak,energy)

    energy = np.asarray(energy, dtype=float)
    safe_energy = np.maximum(energy, 1e-12)

    gaussian = gaussian_peak * np.exp(-0.5 * ((safe_energy - mean_energy) / sigma)**2)

    tail = tail_A * safe_energy**(-tail_alpha) * np.exp(-safe_energy / tail_E0)

    flux = gaussian + tail


    return flux

# create log-prior, given parameters theta and the neutron flux model
# should return log-prior distribution for the entire group structure
def log_prior(theta,model): 

    # unpack the tuple of parameters
    sigma, gaussian_peak, tail_A, tail_alpha, tail_E0 = theta

    # set hard limits for the parameters
    if (
        0.1 < sigma < 2
        and 1e3 < gaussian_peak < 1e9
        and -1e6 < tail_A < 1e6
        and -10 < tail_alpha < 10
        and 1e-6 < tail_E0 < 1e3
    ):

        # generate and sum model values for the group structure
        # to get the prior
        prior = np.sum([model(theta,i) for i in unfold.group_structure])

        # return log(prior) for the group structure (MeV) if in limits
        return np.log(prior)
    
    # return -inf if outside limits 
    return -np.inf

# grab the log likelihood and log posterior
# these are automatically constructed from the prior/model at runtime
log_likelihood = unfold.log_likelihood
log_posterior = unfold.log_posterior

# number of samples from the response function distributions
# computational expense skyrockets here, so start with 1
rm_samples = 5

# number of MCMC walkers/chains
nwalkers = 20

# "burn-in" period to let chains stabilize
nburn = 50

# total number of MCMC steps to take (including nburn)
# no. of trace results =  nwalkers * (nsteps-nburn)
nsteps = 100

# run sampler on single cpu
#if __name__ == '__main__':
#    samples = unfold.run_sampler(log_posterior,model,log_prior,
#                                 log_likelihood,rm_samples,
#                                 nwalkers,nburn,nsteps)
#    
#    # postprocess and save results
#    params,cov_matrix = unfold.postpro_sampler(samples)
#    np.savetxt('params.txt', np.transpose(params))
#    np.savetxt('cov_matrix.txt', np.transpose(cov_matrix))
#
#    # do lazy stdev calculation and plot spectrum
#    stds = np.diag(np.sqrt(cov_matrix))
#    unfold.plot_simple_spectrum(model,params,stds)

# run sampler in parallel
if __name__ == '__main__':
   with mp.Pool() as pool:
        samples=unfold.run_sampler(log_posterior,model,log_prior,
                                   log_likelihood,rm_samples,
                                   nwalkers,nburn,nsteps,pool)
        
        # postprocess and save results
        params,cov_matrix = unfold.postpro_sampler(samples)
        np.savetxt('params.txt', np.transpose(params))
        np.savetxt('cov_matrix.txt', np.transpose(cov_matrix))

        # lazy stdev calculation and plot spectrum
        stds = np.diag(np.sqrt(cov_matrix))
        unfold.plot_simple_spectrum(model,params,stds)