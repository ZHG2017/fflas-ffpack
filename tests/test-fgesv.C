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
//                        Test for fgesv : 1 computation
//
//--------------------------------------------------------------------------
// Clement Pernet
//-------------------------------------------------------------------------

//#define __FFLASFFPACK_DEBUG 1
#define TIME 1

#include <iomanip>
#include <iostream>
using namespace std;

#include "fflas-ffpack/field/modular-balanced.h"
#include "fflas-ffpack/utils/timer.h"
#include "fflas-ffpack/utils/fflas_io.h"
#include "fflas-ffpack/ffpack/ffpack.h"


using namespace FFPACK;

typedef Givaro::Modular<double> Field;

int main(int argc, char** argv){

	int n,m,mb,nb;
	cerr<<setprecision(10);
	Field::Element zero, one;

	if (argc != 6)	{
		cerr<<"Usage : test-fgesv <p> <A> <b> <iter> <left/right>"
		    <<endl;
		exit(-1);
	}
	int nbit=atoi(argv[4]); // number of times the product is performed
	Field F(atoi(argv[1]));
	F.init(zero,0.0);
	F.init(one,1.0);
	Field::Element * A, *B, *X=NULL;
	FFLAS::ReadMatrix (argv[2],F,m,n,A);
	FFLAS::ReadMatrix (argv[3],F,mb,nb,B);

	FFLAS::FFLAS_SIDE side = (atoi(argv[5])) ? FFLAS::FflasRight :  FFLAS::FflasLeft;

	size_t ldx=0;
	size_t rhs = (side == FFLAS::FflasLeft) ? nb : mb;
	if (m != n) {
		if (side == FFLAS::FflasLeft){
			X = FFLAS::fflas_new<Field::Element>(n*nb);
			ldx = nb;
		}
		else {
			X = FFLAS::fflas_new<Field::Element>(mb*m);
			ldx = m;
		}
	}

	if ( ((side == FFLAS::FflasRight) && (n != nb))
	     || ((side == FFLAS::FflasLeft)&&(m != mb)) ) {
		cerr<<"Error in the dimensions of the input matrices"<<endl;
		exit(-1);
	}
	int info=0;
 FFLAS::Timer t; t.clear();
	double time=0.0;
	//FFLAS::WriteMatrix (cerr<<"A="<<endl, F, k,k,A,k);
	size_t R=0;
	for (int i = 0;i<nbit;++i){
		t.clear();
		t.start();
		if (m == n)
			R = FFPACK::fgesv (F, side, mb, nb, A, n, B, nb, &info);
		else
			R = FFPACK::fgesv (F, side, m, n, rhs, A, n, X, ldx, B, nb, &info);
		if (info > 0){
			std::cerr<<"System is inconsistent"<<std::endl;
			exit(-1);
		}

		t.stop();
		time+=t.usertime();
		if (i+1<nbit){
			FFLAS::fflas_delete(A);
			FFLAS::ReadMatrix (argv[2],F,m,n,A);
			FFLAS::fflas_delete( B);
			FFLAS::ReadMatrix (argv[3],F,mb,nb,B);
		}
	}

#ifdef __FFLASFFPACK_DEBUG
	Field::Element  *B2=NULL;
	FFLAS::fflas_delete( A);

	if (info > 0){
		std::cerr<<"System inconsistent"<<std::endl;
		exit (-1);
	}

	FFLAS::ReadMatrix (argv[2],F,m,n,A);

	B2 = FFLAS::fflas_new<Field::Element>(mb*nb);


	if (m==n)
		if (side == FFLAS::FflasLeft)
			FFLAS::fgemm (F, FFLAS::FflasNoTrans, FFLAS::FflasNoTrans, m, nb, n,
				      one, A, n, B, nb, zero, B2, nb);
		else
			FFLAS::fgemm (F, FFLAS::FflasNoTrans, FFLAS::FflasNoTrans, mb, n, m,
				      one, B, nb, A, n, zero, B2, nb);
	else
		if (side == FFLAS::FflasLeft)
			FFLAS::fgemm (F, FFLAS::FflasNoTrans, FFLAS::FflasNoTrans, m, nb, n,
				      one, A, n, X, ldx, zero, B2, nb);
		else
			FFLAS::fgemm (F, FFLAS::FflasNoTrans, FFLAS::FflasNoTrans, mb, n, m,
				      one, X, ldx, A, n, zero, B2, nb);
	FFLAS::fflas_delete( B);
	FFLAS::fflas_delete( X);

	FFLAS::ReadMatrix (argv[3],F,mb,nb,B);

	bool wrong = false;
	for (int i=0;i<mb;++i)
		for (int j=0;j<nb;++j)
			if ( !F.areEqual(*(B2+i*nb+j), *(B+i*nb+j))){
				cerr<<"B2 ["<<i<<", "<<j<<"] = "<<(*(B2+i*nb+j))
				    <<" ; B ["<<i<<", "<<j<<"] = "<<(*(B+i*nb+j))
				    <<endl;
				wrong = true;
			}

	if (wrong) {
		cerr<<"FAIL"<<endl;
		    //FFLAS::WriteMatrix (cerr<<"B2="<<endl,F,m,n,B2,n);
		    //FFLAS::WriteMatrix (cerr<<"B="<<endl,F,m,n,B,n);
	}else{

		cerr<<"PASS"<<endl;
	}


	FFLAS::fflas_delete( B2);
#endif

	FFLAS::fflas_delete( A);
	FFLAS::fflas_delete( B);
#if TIME
	double mflops;
	double cplx = (double)n*m*m-(double)m*m*m/3;
	if (side == FFLAS::FflasLeft)
		mflops = (cplx+(double)(2*R*R*n))/1000000.0*nbit/time;
	else
		mflops = (cplx+(double)(2*R*R*m))/1000000.0*nbit/time;
	cerr<<"m,n,mb,nb = "<<m<<" "<<n<<" "<<mb<<" "<<nb<<". fgesv "
	    <<((side == FFLAS::FflasLeft)?" Left ":" Right ")
	    <<"over Z/"<<atoi(argv[1])<<"Z :"
	    <<endl
	    <<"t= "
	    << time/nbit
	    << " s, Mffops = "<<mflops
	    << endl;

	cout<<m<<" "<<n<<" "<<mflops<<" "<<time/nbit<<endl;
#endif
}
