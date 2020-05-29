cd ParticleFilter
R --vanilla --silent -e 'Rcpp::compileAttributes()'
cd ..
R CMD build ParticleFilter
R CMD INSTALL ParticleFilter*.tar.gz

