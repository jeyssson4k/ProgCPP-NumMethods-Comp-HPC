/**
* PDF for the normal distribution
* @author Jeysson4K
* @workshop 02
* @course INTRO HPC
**/

#include <random>
#include <iostream>
#include <cstdlib>
#include <vector>

void compute_pdf(int seed, int nsamples, double mu, double sigma, double xmin, double xmax, int nbins);

int main(int argc, char **argv){
  const int SEED = std::atoi(argv[1]);
  const int NSAMPLES = std::atoi(argv[2]);
  const double MU = std::atof(argv[3]);
  const double SIGMA = std::atof(argv[4]);
  const double XMIN = std::atof(argv[5]);
  const double XMAX = std::atof(argv[6]);
  const int NBINS = std::atoi(argv[7]);
    
  compute_pdf(SEED, NSAMPLES, MU, SIGMA, XMIN, XMAX, NBINS);
  return 0;
}

void compute_pdf(int seed, int nsamples, double mu, double sigma, double xmin, double xmax, int nbins){
  // random stuff
  std::mt19937 gen(seed); //creates the specific seed 
  std::normal_distribution<double> dis(mu, sigma); //initializes the normal distribution 

  // histogram stuff
  std::vector<int> histogram(nbins, 0); //sets size = nbins, initializes its value in 0
  double binWidth = (xmax - xmin) / nbins; //calculates bin size 
  
  //fills the histogram
  for (int n = 0; n < nsamples; ++n) {
    double r = dis(gen); //generates a random number based on the normal distribution 
    
    //the number is out of range then ignore it 
    if (r < xmin || r >= xmax) {
        continue;
    }
    int binIndex = static_cast<int>((r - xmin) / binWidth); //defines what bin the number is in 
    histogram[binIndex]++; //increments bin counter
  }

  // compute and print the pdf
  for (int i = 0; i < nbins; ++i) {
    double zeroCenter = 0.5 * binWidth, centerStep = i * binWidth; //variables for mid value
    double binCenter = xmin + zeroCenter + centerStep; //calculates mid value for each bin
    double pdf = static_cast<double>(histogram[i]) / (nsamples * binWidth); //frecuency of intervals

    std::printf("%.5f %.5f\n", binCenter, pdf); //output 
  }
}
/*
* https://cplusplus.com/reference/random/mersenne_twister_engine/mersenne_twister_engine/
* https://en.cppreference.com/w/cpp/language/static_cast
* https://matplotlib.org/stable/api/_as_gen/matplotlib.markers.MarkerStyle.html
* https://en.wikipedia.org/wiki/Probability_density_function
* https://en.cppreference.com/w/cpp/numeric/random
* https://learn.microsoft.com/es-es/cpp/cpp/static-cast-operator?view=msvc-170
* https://en.cppreference.com/w/cpp/numeric/random/normal_distribution
* https://en.cppreference.com/w/cpp/named_req/RandomNumberDistribution
* https://en.cppreference.com/w/cpp/container/vector
* https://learn.microsoft.com/es-es/cpp/cpp/constexpr-cpp?view=msvc-170
* https://en.cppreference.com/w/cpp/container/vector/vector
* https://www-h.eng.cam.ac.uk/help/tpl/languages/C++/casting.html
*/