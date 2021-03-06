/* -*- mode: C++; tab-width: 8; indent-tabs-mode: t; c-basic-offset: 8 -*- */
// vim:sts=8:sw=8:ts=8:noet:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s

/*
 * Copyright (C) FFLAS-FFPACK
 * Written by Clément Pernet
 * This file is Free Software and part of FFLAS-FFPACK.
 *
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
 *.
 */


//--------------------------------------------------------------------------
//                        Test for rank
//
//--------------------------------------------------------------------------
// Clement Pernet
//-------------------------------------------------------------------------

#include <iomanip>
#include <iostream>
#include "givaro/modular-balanced.h"
#include "fflas-ffpack/utils/timer.h"
#include "fflas-ffpack/utils/fflas_io.h"
#include "fflas-ffpack/ffpack/ffpack.h"



using namespace std;
using namespace FFPACK;

typedef ModularBalanced<double> Field;

int main(int argc, char** argv){

	int n,m;
	int nbit=atoi(argv[3]); // number of times the product is performed
	cerr<<setprecision(10);
	if (argc !=  4)	{
		cerr<<"Usage : test-rank <p> <A> <<i>"
		    <<endl
		    <<"         to compute the rank of A mod p (i computations)"
		    <<endl;
		exit(-1);
	}
	Field F(atof(argv[1]));
	Field::Element * A;
	FFLAS::ReadMatrix (argv[2],F,m,n,A);

 FFLAS::Timer tim,t;
	t.clear();
	tim.clear();
	size_t r=0;
	for(int i = 0;i<nbit;++i){
		t.clear();
		t.start();
		r = FFPACK::Rank (F, m, n, A, n);
		t.stop();
		tim+=t;
		if (i+1<nbit){
			FFLAS::fflas_delete( A);
			FFLAS::ReadMatrix (argv[2],F,m,n,A);
		}
	}

	double mflops = 2.0/3.0*(n*(double)r/1000000.0)*nbit*n/tim.usertime();
	cerr<<"m,n = "<<m<<", "<<n<<" Rank (A) = " << r
		     << " mod "<<atoi(argv[1])<<" : t= "
		     << tim.usertime()/nbit
		     << " s, Mffops = "<<mflops
		     << endl;

	cout<<m<<" "<<n<<" "<<r<<" "<<mflops<<" "<<tim.usertime()/nbit<<endl;
}
