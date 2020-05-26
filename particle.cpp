// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
using namespace arma;
using namespace Rcpp;


void resampleParticles(Rcpp::NumericMatrix& rParticles,
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

void initialiseVariables(Rcpp::NumericMatrix& resampledParticles, Rcpp::NumericVector& initialState){
  for(int col = 0 ; col < resampledParticles.ncol(); col++) {
      resampledParticles(_,col) = initialState;
  }
}

void simulateTransition(Rcpp::NumericMatrix::Column particle,
                        Rcpp::NumericMatrix::Column rParticle,
                        Rcpp::NumericVector& params)
  {
  int ntrans = (int) 1 / params[1];
  int S = rParticle[0];
  int I = rParticle[1];
  int R = rParticle[2];

  for(int i = 0 ; i < ntrans ; i++) {
    double p_SI = 1 - exp(-(params[2] * I * params[1])/params[0]);
    double p_IR = 1 - exp(-params[4] * params[1]);

    int dN_SI = R::rbinom(S, p_SI);
    int dN_IR = R::rbinom(I, p_IR);

    S -= dN_SI;
    I += (dN_SI - dN_IR);
    R += dN_IR;
  }

  particle[0] = S;
  particle[1] = I;
  particle[2] = R;
}

void propagateParticles(Rcpp::NumericMatrix& particles,
                        Rcpp::NumericMatrix& resampledParticles,
                        Rcpp::NumericVector& params){
  //loop through each particle and run single timestep obs
  for(int i = 0; i < particles.ncol(); i++) {
      simulateTransition(particles(_,i), resampledParticles(_,i), params);
  }
}

void weightParticles(double y,
                     Rcpp::NumericVector& weights,
                     Rcpp::NumericMatrix& particles,
                     Rcpp::NumericVector& params){
  for(int i = 0; i < weights.size(); i++) {
    Rcpp::NumericMatrix::Column states = particles(_,i);
    weights[i] = R::dpois(y, params[3] * states[1], 0);
    //  if(weights[i]==0) {
    //  Rcout << "----" << "\n";
    //  Rcout << "rho is: " << params[3] << "\n";
    //  Rcout << "I is: " << states[1] << "\n";
    //  Rcout << "y is: " << y << "\n"; 
    //  Rcout << "----" << "\n";
    //  }
  }
}

double computeLikelihood(Rcpp::NumericVector& weights){
  int N = weights.size();
  return log(sum(weights)/N);
}

// Particle filter for sir model
// Param vectors take form [N, dt, beta, rho, gamma]
// State vectors take  form [S, I , R]
// [[Rcpp::export(name=particleFilterCpp)]]
double particleFilter(Rcpp::NumericVector y,
                    Rcpp::NumericVector params,
                    int n_particles,
                    Rcpp::NumericVector initialState
                    ) {

  // Initialising
  // Marshalling
  double logLikelihood = 0;
  int n_obs = y.size();

  int num_transitions = 1 / params[1];

  // Create Procedure Variables
  Rcpp::NumericMatrix particles(3, n_particles);
  Rcpp::NumericMatrix resampledParticles(3, n_particles);
  Rcpp::NumericVector weights(n_particles);
  initialiseVariables(resampledParticles, initialState);

  for(int t = 0; t < n_obs; t++) {
    propagateParticles(particles, resampledParticles, params);
    weightParticles(y[t], weights, particles, params);
    resampleParticles(resampledParticles, particles, weights);
    logLikelihood += computeLikelihood(weights);
  }

  return logLikelihood;
}
