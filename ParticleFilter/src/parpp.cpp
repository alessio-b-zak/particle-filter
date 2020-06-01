#include <Rcpp.h>
#include <RcppParallel.h>
#include <sitmo.h>
#include <omp.h>
#include <boost/random/binomial_distribution.hpp>
using namespace Rcpp;
using namespace RcppParallel;
using binomial = boost::random::binomial_distribution<int>;


// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(openmp)]]
// [[Rcpp::depends(sitmo)]]

void parResampleParticles(Rcpp::NumericMatrix& rParticles,
                       Rcpp::NumericMatrix& particles,
                       Rcpp::NumericVector& weights){
  // Rcout << "num particles: " << particles.ncol() << "\n";
  // int zcount = 0;
  // for(int i = 0; i < particles.ncol() ; i++) {
  //   if(weights[i] == 0) {
  //     zcount++;
  //   }
  // }
  // Rcout << "num zero weights: " << zcount << "\n";
  Rcpp::IntegerVector indices = Rcpp::sample(particles.ncol(),particles.ncol(), true, weights, false);
  for(int i = 0; i < particles.ncol() ; i++) {
    rParticles(_, i) = particles(_,indices(i));
  }
}

void parInitialiseVariables(Rcpp::NumericMatrix& resampledParticles, Rcpp::NumericVector& initialState){
  for(int col = 0 ; col < resampledParticles.ncol(); col++) {
      resampledParticles(_,col) = initialState;
  }
}

void parSimulateTransition(RcppParallel::RMatrix<double>::Column particle,
                        RcppParallel::RMatrix<double>::Column rParticle,
                        RcppParallel::RVector<double> params,
                        sitmo::prng_engine& engine,
                        binomial& transitions)
  {
  int ntrans = (int) 1 / params[1];
  int S = rParticle[0];
  int I = rParticle[1];
  int R = rParticle[2];

  for(int i = 0 ; i < ntrans ; i++) {
    double p_SI = 1 - exp(-(params[2] * I * params[1])/params[0]);
    double p_IR = 1 - exp(-params[4] * params[1]);

    int dN_SI = transitions(engine, binomial::param_type(S, p_SI));
    int dN_IR = transitions(engine, binomial::param_type(I, p_IR));

    S -= dN_SI;
    I += (dN_SI - dN_IR);
    R += dN_IR;
  }


  particle[0] = S;
  particle[1] = I;
  particle[2] = R;
}

void parNormaliseWeights(Rcpp::NumericVector& tWeights,
                       Rcpp::NumericVector& nWeights) {
  nWeights = exp(tWeights) / sum(exp(tWeights));
}

void parPropagateParticles(RcppParallel::RMatrix<double>& particles,
                        RcppParallel::RMatrix<double>& resampledParticles,
                        RcppParallel::RVector<double>& params,
                        std::vector<sitmo::prng_engine>& engines,
                        binomial& transitions,
                        int ncores){
  //loop through each particle and run single timestep obs
  #pragma omp parallel num_threads(ncores)
  {
    #pragma omp for
    for(int i = 0; i < particles.ncol(); i++) {
        parSimulateTransition(particles.column(i), 
        resampledParticles.column(i), 
        params, 
        engines[omp_get_thread_num()], 
        transitions);
    }
  }
}

void parWeightParticles(double y,
                     Rcpp::NumericVector& weights,
                     Rcpp::NumericMatrix& particles,
                     Rcpp::NumericVector& params
                     ){
  for(int i = 0; i < weights.size(); i++) {
    Rcpp::NumericMatrix::Column states = particles(_,i);
    weights[i] = R::dpois(y, params[3] * states[1], 1);

    //  if(weights[i]==0) {
    //  Rcout << "----" << "\n";
    //  Rcout << "rho is: " << params[3] << "\n";
    //  Rcout << "I is: " << states[1] << "\n";
    //  Rcout << "y is: " << y << "\n"; 
    //  Rcout << "----" << "\n";
    //  }
  }
}

double parComputeLikelihood(Rcpp::NumericVector& tWeights, double lWeightsMax, int n_particles){
  double tLikelihood = lWeightsMax + log(sum(exp(tWeights))) - log(n_particles);
  return tLikelihood;
}

// Particle filter for sir model
// Param vectors take form [N, dt, beta, rho, gammalogiiikjkjik]
// State vectors take  form [S, I , R]
// [[Rcpp::export(name=parParticleFilterCpp)]]
double parParticleFilter(Rcpp::NumericVector y,
                    Rcpp::NumericVector params,
                    int n_particles,
                    Rcpp::NumericVector initialState,
                    int ncores
                    ) {

  // Initialising
  // Marshalling
  double logLikelihood = 0;
  double otherLikelihood = 0;
  int n_obs = y.size();

  int num_transitions = 1 / params[1];

  // Create Procedure Variables
  Rcpp::NumericMatrix particles(3, n_particles);
  RcppParallel::RMatrix<double> oParticles(particles);

  Rcpp::NumericMatrix resampledParticles(3, n_particles);
  RcppParallel::RMatrix<double> oRParticles(resampledParticles);

  Rcpp::NumericVector weights(n_particles);
  RcppParallel::RVector<double> oWeights(weights);

  Rcpp::NumericVector nWeights(n_particles);
  RcppParallel::RVector<double> oNWeights(weights);
  
  Rcpp::NumericVector tWeights(n_particles);
  RcppParallel::RVector<double> oTWeights(weights);

  RcppParallel::RVector<double> oParams(params);

  double lWeightsMax;

  RNGScope scope;

  NumericVector seeds(ncores);

  std::vector<sitmo::prng_engine> engArray(ncores);

  seeds[0] = runif(1, 1.0, std::numeric_limits<uint32_t>::max())[0];

  for (unsigned int j = 0;  j < ncores - 1; j++){
    seeds[j+1] = seeds[j] - 1.0;
    if (seeds[j+1] < 1.0) seeds[j] = std::numeric_limits<uint32_t>::max() - 1.0;
    engArray[j] = sitmo::prng_engine(static_cast<uint32_t>(seeds[j]));
  }

  binomial transitions;

  parInitialiseVariables(resampledParticles, initialState);

  for(int t = 0; t < n_obs; t++) {
    parPropagateParticles(oParticles, oRParticles, oParams, engArray, transitions, ncores);

    parWeightParticles(y[t], weights, particles, params );
    lWeightsMax = Rcpp::max(weights);
    tWeights = weights - lWeightsMax;

    parNormaliseWeights(tWeights, nWeights);
    parResampleParticles(resampledParticles, particles, nWeights);
    double ll = parComputeLikelihood(tWeights, lWeightsMax, n_particles);
    logLikelihood += ll;
  }


  return logLikelihood;
}
