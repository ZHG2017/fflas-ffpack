/* Copyright (c) FFLAS-FFPACK
* ========LICENCE========
* This file is part of the library FFLAS-FFPACK.
*
* FFLAS-FFPACK is free software: you can redistribute it and/or modify
* it under the terms of the  GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with this library; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
* ========LICENCE========
*/


#include "fflas-ffpack/fflas/fflas.h"

#include <iostream>

#include "fflas-ffpack/utils/timer.h"
#include "fflas-ffpack/utils/args-parser.h"

#ifdef __FFLASFFPACK_USE_KAAPI
#include "libkomp.h"
#endif

namespace FFLAS
{
  template <typename RandIter, typename RNS>
  void
  frand (const FFPACK::RNSInteger<RNS> &F, RandIter &G, const size_t m,
         const size_t n, typename FFPACK::RNSInteger<RNS>::Element_ptr A,
         const size_t lda)
  {
    for (size_t i = 0; i < m; i++)
    {
      for (size_t j = 0; j < m; j++)
      {
        G.random(A[j+i*lda]);
      }
    }
  }

  template <typename RandIter, typename RNS>
  void
  fzero (const FFPACK::RNSInteger<RNS> &F, const size_t m, const size_t n,
         typename FFPACK::RNSInteger<RNS>::Element_ptr A, const size_t lda)
  {
    for (size_t i = 0; i < m; i++)
    {
      for (size_t j = 0; j < m; j++)
      {
        A[j+i*lda].assign(F.zero);
      }
    }
  }
}

using namespace FFLAS;

int
main(int argc, char *argv[])
{
  size_t pbits = 20;
  size_t r = 8;
  size_t m = 1000;
  size_t k = 1000;
  size_t n = 1000;
  int nbw = -1;
  size_t iter = 3;
  int fgemm_th = MAX_THREADS;
  int moduli_th = 1;

  Argument as[] = {
    { 'r', "-r R", "Number of RNS moduli (default is 8).", TYPE_INT , &r},
    { 'b', "-b B", "Number of bits of the moduli (in [10, 26], default is 20)",
                                                              TYPE_INT, &pbits},
    { 'm', "-m M", "Row dimension of A (default is 1000).", TYPE_INT, &m},
    { 'k', "-k K", "Col dimension of A (default is 1000).", TYPE_INT, &k},
    { 'n', "-n N", "Col dimension of B (default is 1000).", TYPE_INT, &n},
    { 'w', "-w N", "Number of Winograd levels (default is -1, means random).",
                                                                TYPE_INT, &nbw},
    { 'i', "-i R", "Number of repetitions (default is 3).", TYPE_INT, &iter},
    { 't', "-t T", "Number of threads for the fgemm computations.", TYPE_INT,
                                                                     &fgemm_th},
    { 'u', "-u U", "Number of \"threads\" for handling the moduli.", TYPE_INT,
                                                                    &moduli_th},
    END_OF_ARGUMENTS
  };

  parseArguments (argc, argv, as);

  /* Check some command line parameters */
  if (fgemm_th <= 0 || moduli_th <= 0)
  {
    std::cerr << "Error, the number of threads must be positive" << std::endl;
    return 1;
  }
  if (pbits < 10 || pbits > 26)
  {
    std::cerr << "Error, the number of bits of the moduli must be in [10, 26]"
              << std::endl;
    return 1;
  }

  typedef FFPACK::rns_double RNS;
  typedef FFPACK::RNSInteger<RNS> Field;
  typedef Field::Element_ptr Element_ptr;

  RNS rns (pbits, r);
  Field ZZ(rns);

  Timer chrono, TimFreivalds;
  double time=0.0, timev=0.0;

  Element_ptr A, B, C;

  Field::RandIter G(ZZ);

  A = fflas_new (ZZ, m, k, Alignment::CACHE_PAGESIZE);
  frand (ZZ, G, m, k, A, k);
  B = fflas_new (ZZ, k, n, Alignment::CACHE_PAGESIZE);
  frand (ZZ, G, k, n, B, n);
  C = fflas_new (ZZ, m, n, Alignment::CACHE_PAGESIZE);
  fzero (ZZ, m, n, C, n);

  for (size_t i=0; i<=iter; ++i)
  {
    chrono.clear();
    if (i)
      chrono.start();
              
    PAR_BLOCK
    {
      typedef ParSeqHelper::Parallel<> PPar;
      typedef ParSeqHelper::RNSParallel<PPar, PPar> RNSParPar;
      typedef MMHelper<Field, MMHelperAlgo::Winograd, ModeCategories::DefaultTag, RNSParPar> RNSHelper;
      RNSParPar RNSPP (moduli_th, fgemm_th);
      RNSHelper WH (ZZ, nbw, RNSPP);
      fgemm (ZZ, FflasNoTrans, FflasNoTrans, m,n,k, ZZ.one, A, k, B, n, ZZ.zero, C,n, WH);
    }

    if (i)
    {
      chrono.stop();
      time+=chrono.realtime();
    }

    TimFreivalds.clear();
    TimFreivalds.start();
      
    bool pass = freivalds (ZZ, FflasNoTrans, FflasNoTrans, m, n, k, ZZ.one, A, k, B, n, C,n);
    TimFreivalds.stop();
    timev += TimFreivalds.usertime();
    if (!pass) 
      std::cout << "FAILED" << std::endl;
  }
  
  fflas_delete (A);
  fflas_delete (B);
  fflas_delete (C);
  
  // -----------
  // Standard output for benchmark
  double time_per_iter = time / double(iter);
  double Gflops = double(iter) * double (r) * (2. * double(m)/1000. * double(n)/1000. * double(k)/1000.0) / time;
  std::cout << "Time: " << time_per_iter << " Gflops: " << Gflops;
  writeCommandString (std::cout, as) << std::endl;
#if __FFLASFFPACK_DEBUG
  std::cout << "Freivalds vtime: " << timev / (double)iter << std::endl;
#endif

  return 0;
}

