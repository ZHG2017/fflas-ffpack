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
//                        Test for nullspace
//
//--------------------------------------------------------------------------
// Clement Pernet
//-------------------------------------------------------------------------

#define TIME 1
using namespace std;

#include <iomanip>
#include <iostream>
#include "fflas-ffpack/field/modular-balanced.h"
#include "fflas-ffpack/utils/timer.h"
#include "fflas-ffpack/utils/fflas_io.h"
#include "fflas-ffpack/ffpack/ffpack.h"


using namespace FFPACK;
typedef ModularBalanced<double> Field;

int main(int argc, char** argv){

	int n,m;
	int nbit=atoi(argv[3]); // number of times the product is performed
	cerr<<setprecision(10);
	Field::Element zero, one;

	if (argc != 4)	{
		cerr<<"Usage : test-nullspace <p> <A> <<i>"
		    <<endl
		    <<"         to compute the nullspace of A mod p (i computations)"
		    <<endl;
		exit(-1);
	}
	Field F(atof(argv[1]));
	F.init(zero,0.0);
	F.init(one,1.0);
	Field::Element * A, *NS;
	FFLAS::ReadMatrix (argv[2],F,m,n,A);

	FFLAS::Timer tim,t; t.clear();tim.clear();
	size_t  ldn, NSdim;

	for(int i = 0;i<nbit;++i){
		t.clear();
		t.start();
		FFPACK::NullSpaceBasis (F, FFLAS::FflasRight, m,n,
					A, n, NS, ldn, NSdim);
// 		FFPACK::NullSpaceBasis (F, FFLAS::FflasLeft, m,n,
// 					A, n, NS, ldn, NSdim);
		t.stop();
		tim+=t;
	}

#if __FFLASFFPACK_DEBUG
	FFLAS::ReadMatrix (argv[2],F,m,n,Ab);
	Field::Element *C = FFLAS::fflas_new<Field::Element>(NSdim*n);
 	FFLAS::fgemm (F, FFLAS::FflasNoTrans, FFLAS::FflasNoTrans, m, NSdim, n,
 		      1.0, Ab, n, NS, ldn, 0.0, C, NSdim);
// 	FFLAS::fgemm (F, FFLAS::FflasNoTrans, FFLAS::FflasNoTrans, NSdim, n, m,
// 		      1.0, NS, ldn, Ab, n, 0.0, C, n);
	bool wrong = false;

	for (int i=0;i<m;++i)
		for (size_t j=0;j<NSdim;++j)
			if (!F.areEqual(*(C+i*NSdim+j),zero))
				wrong = true;
// 	for (int i=0;i<NSdim;++i)
// 		for (int j=0;j<n;++j)
// 			if (!F.areEqual(*(C+i*n+j),zero))
// 				wrong = true;

	if ( wrong ){
		cerr<<"FAIL"<<endl;
		FFLAS::WriteMatrix (cerr<<"A="<<endl,F,m,n,Ab,n);
		FFLAS::WriteMatrix (cerr<<"NS="<<endl,F, n,NSdim, NS, NSdim);
		FFLAS::WriteMatrix (cerr<<"C="<<endl,F,m,NSdim, C, NSdim);
	} else {
		cerr<<"PASS"<<endl;
	}
	FFLAS::fflas_delete( C);
	FFLAS::fflas_delete( Ab);

#endif
	FFLAS::fflas_delete( NS);
	FFLAS::fflas_delete( A);

#if TIME
	double mflops = 2*(n*n/1000000.0)*nbit*n/tim.usertime();
	cerr<<"NSdim = "<<NSdim<<" Nullspace over Z/"<<atoi(argv[1])<<"Z : t= "
	     << tim.usertime()/nbit
	     << " s, Mffops = "<<mflops
	     << endl;

	cout<<n<<" "<<mflops<<" "<<tim.usertime()/nbit<<endl;
#endif
}
