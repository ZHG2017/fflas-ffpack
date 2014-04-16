/* -*- mode: C++; tab-width: 8; indent-tabs-mode: t; c-basic-offset: 8 -*- */
// vim:sts=8:sw=8:ts=8:noet:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s
/* fflas/fflas_finit.inl
 * Copyright (C) 2014 Pascal Giorgi
 *
 * Written by Pascal Giorgi <Pascal.Giorgi@lirmm.fr>
 * BB<bboyer@ncsu.edu>
 *
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
#ifndef __FFLASFFPACK_fflas_avx_functions_H
#define __FFLASFFPACK_fflas_avx_functions_H

// #include <x86intrin.h>
#include <immintrin.h>

#define DEBUG_M256(M) \
{                    \
	float ssssssss[8] ; \
	_mm256_store_ps(ssssssss,M); \
	for (size_t ii = 0 ; ii < 8 ; ++ii) \
	std::cout << ssssssss[ii] << ", " ; \
	std::cout << std::endl; \
}


// FMOD
namespace FFLAS { namespace vectorised { /*  FMOD */

#if defined(__AVX2__) || defined(__AVX__)
#if defined(__AVX2__)

	// compute C modulo P in the range [MIN, MAX]
#define VEC_MODF_D(C,Q,P,NEGP,INVP,T,MIN,MAX)				\
	{								\
		Q = _mm256_mul_pd(C,INVP);                              \
		Q = _mm256_floor_pd(Q);                                 \
		C = _mm256_fnmadd_pd(Q,P,C);                            \
		Q = _mm256_cmp_pd(C,MAX,_CMP_GT_OS);			\
		T = _mm256_cmp_pd(C,MIN,_CMP_LT_OS);			\
		Q = _mm256_and_pd(Q,NEGP);				\
		T = _mm256_and_pd(T,P);					\
		Q = _mm256_or_pd(Q,T);				        \
		C = _mm256_add_pd(C,Q);					\
	}

	// compute C modulo P in the range [MIN, MAX]
#define VEC_MODF_S(C,Q,P,NEGP,INVP,T,MIN,MAX)				\
	{								\
		Q = _mm256_mul_ps(C,INVP);                              \
		Q = _mm256_floor_ps(Q);	                                \
		C = _mm256_fnmadd_ps(Q,P,C);				\
		Q = _mm256_cmp_ps(C,MAX,_CMP_GT_OS);			\
		T = _mm256_cmp_ps(C,MIN,_CMP_LT_OS);			\
		Q = _mm256_and_ps(Q,NEGP);				\
		T = _mm256_and_ps(T,P);					\
		Q = _mm256_or_ps(Q,T);				        \
		C = _mm256_add_ps(C,Q);					\
	}

	// compute C modulo P in the range [O, P-1] (not sure faster)
#define VEC_MODF_D_POS(C,Q,P,INVP,T)                  \
	{       Q = _mm256_mul_pd(C,INVP);            \
		Q = _mm256_floor_pd(Q);               \
		C = _mm256_fnmadd_pd(Q,P,C);          \
		Q = C;                                \
		Q = _mm256_cmp_pd(Q,P,_CMP_GE_OS);    \
		Q = _mm256_and_pd(Q,P);               \
		C = _mm256_sub_pd(C,Q);               \
	}

	// compute C modulo P in the range [O, P-1] (not sure faster)
#define VEC_MODF_S_POS(C,Q,P,INVP,T)                  \
	{       Q = _mm256_mul_ps(C,INVP);            \
		Q = _mm256_floor_ps(Q);               \
		C = _mm256_fnmadd_ps(Q,P,C);          \
		Q = C;                                \
		Q = _mm256_cmp_ps(Q,P,_CMP_GE_OS);    \
		Q = _mm256_and_ps(Q,P);               \
		C = _mm256_sub_ps(C,Q);               \
	}


#else //  defined(__AVX__)

#define VEC_MODF_D(C,Q,P,NEGP,INVP,T,MIN,MAX)				\
	{								\
		Q = _mm256_mul_pd(C,INVP);                              \
		Q = _mm256_floor_pd(Q);                                 \
		T = _mm256_mul_pd(Q,P);                                 \
		C = _mm256_sub_pd(C,T);		                        \
		Q = _mm256_cmp_pd(C,MAX,_CMP_GT_OS);			\
		T = _mm256_cmp_pd(C,MIN,_CMP_LT_OS);			\
		Q = _mm256_and_pd(Q,NEGP);				\
		T = _mm256_and_pd(T,P);					\
		Q = _mm256_or_pd(Q,T);				        \
		C = _mm256_add_pd(C,Q);					\
	}


#define VEC_MODF_S(C,Q,P,NEGP,INVP,T,MIN,MAX)				\
	{								\
		Q = _mm256_mul_ps(C,INVP);                              \
		Q = _mm256_floor_ps(Q);	                                \
		T = _mm256_mul_ps(Q,P);                                 \
		C = _mm256_sub_ps(C,T);	                                \
		Q = _mm256_cmp_ps(C,MAX,_CMP_GT_OS);			\
		T = _mm256_cmp_ps(C,MIN,_CMP_LT_OS);			\
		Q = _mm256_and_ps(Q,NEGP);				\
		T = _mm256_and_ps(T,P);					\
		Q = _mm256_or_ps(Q,T);				        \
		C = _mm256_add_ps(C,Q);					\
	}

	// compute C modulo P in the range [O, P-1] (not sure faster)
#define VEC_MODF_D_POS(C,Q,P,INVP,T)                  \
	{       Q = _mm256_mul_pd(C,INVP);            \
		Q = _mm256_floor_pd(Q);               \
		T = _mm256_mul_pd(Q,P);               \
		C = _mm256_sub_pd(C,T);		      \
		Q = C;                                \
		Q = _mm256_cmp_pd(Q,P,_CMP_GE_OS);    \
		Q = _mm256_and_pd(Q,P);               \
		C = _mm256_sub_pd(C,Q);               \
	}

	// compute C modulo P in the range [O, P-1] (not sure faster)
#define VEC_MODF_S_POS(C,Q,P,INVP,T)                  \
	{       Q = _mm256_mul_ps(C,INVP);            \
		Q = _mm256_floor_ps(Q);               \
		T = _mm256_mul_ps(Q,P);               \
		C = _mm256_sub_ps(C,T);		      \
		Q = C;                                \
		Q = _mm256_cmp_ps(Q,P,_CMP_GE_OS);    \
		Q = _mm256_and_ps(Q,P);               \
		C = _mm256_sub_ps(C,Q);               \
	}

#endif // AVX and AVX2

	template<bool positive> // no default argument unless C++11
	inline void modp( double *T, const double * U, size_t n, double p, double invp, double min, double max)
	{
		register __m256d C,Q,P,NEGP,INVP,TMP,MIN,MAX;
		P   = _mm256_set1_pd(p);
		NEGP= _mm256_set1_pd(-p);
		INVP= _mm256_set1_pd(invp);
		MIN = _mm256_set1_pd(min);
		MAX = _mm256_set1_pd(max);
		long st=long(T)%32;
		size_t i=0;
		if (n < 4) {
			for (;i<n;i++){
				T[i]=fmod(U[i],p);
				if (!positive)
				       	T[i]-=(T[i]>max)?p:0;
				T[i]+=(T[i]<min)?p:0;
			}
			return;

		}
		if (st){ // the array T is not 32 byte aligned (process few elements s.t. (T+i) is 32 bytes aligned)
			for (size_t j=(size_t)st;j<32;j+=8,i++){
				T[i]=fmod(U[i],p);
				if (!positive)
					T[i]-=(T[i]>max)?p:0;
				T[i]+=(T[i]<min)?p:0;
			}
		}
		FFLASFFPACK_check((long(T+i)%32==0));
		if ((long(U+i)%32==0)) {
			// perform the loop using 256 bits SIMD
			for (;i<=n-4;i+=4){
				C=_mm256_load_pd(U+i);
				if (positive) {
					VEC_MODF_D_POS(C,Q,P,INVP,TMP);
				}
				else {
					VEC_MODF_D(C,Q,P,NEGP,INVP,TMP,MIN,MAX);
				}
				_mm256_store_pd(T+i,C);
			}
		}
		// perform the last elt from T without SIMD
		for (;i<n;i++){
			T[i]=fmod(U[i],p);
			if (!positive)
			       	T[i]-=(T[i]>max)?p:0;
			T[i]+=(T[i]<min)?p:0;
		}
	}

	template<bool positive>
	inline void modp( float *T, const float * U,  size_t n, float p, float invp, float min, float max)
	{
		register __m256 C,Q,P,NEGP,INVP,TMP,MIN,MAX;
		P   = _mm256_set1_ps(p);
		NEGP= _mm256_set1_ps(-p);
		INVP= _mm256_set1_ps(invp);
		MIN= _mm256_set1_ps(min);
		MAX= _mm256_set1_ps(max);
		long st=long(T)%32;
		size_t i=0;;
		if (n < 8) {
			for (;i<n;i++){
				T[i]=fmodf(U[i],p);
				if (!positive)
				       	T[i]-=(T[i]>max)?p:0;
				T[i]+=(T[i]<min)?p:0;
			}
			return;
		}
		if (st){ // the array T is not 32 byte aligned (process few elements s.t. (T+i) is 32 bytes aligned)
			for (size_t j=(size_t)st;j<32;j+=4,i++){
				T[i]=fmodf(U[i],p);
				if (!positive)
					T[i]-=(T[i]>max)?p:0;
				T[i]+=(T[i]<min)?p:0;
			}
		}

		FFLASFFPACK_check((long(T+i)%32==0));
		if ((long(U+i)%32==0)) {
			// perform the loop using 256 bits SIMD
			for (;i<=n-8;i+=8){
				C=_mm256_load_ps(U+i);
				if (positive) {
					VEC_MODF_S_POS(C,Q,P,INVP,TMP);
				}
				else {
					VEC_MODF_S(C,Q,P,NEGP,INVP,TMP,MIN,MAX);
				}
				_mm256_store_ps(T+i,C);
			}
		}
		// perform the last elt from T without SIMD
		for (;i<n;i++){
			T[i]=fmodf(U[i],p);
			if (!positive)
				T[i]-=(T[i]>max)?p:0;
			T[i]+=(T[i]<min)?p:0;
		}
	}

#else // no AVX

	template<bool positive>
	inline void modp( double *T, const double * U, size_t n, double p, double invp, double min, double max)
	{
		for(size_t j=0;j<n;j++){
			T[j]= fmod(U[j],p);
			if (!positive)
				T[j]-=(T[j]>max)?p:0;
			T[j]+=(T[j]<min)?p:0;
		}

	}

	template<bool positive>
	inline void modp( float *T, const float * U, size_t n, float p, float invp, float min, float max)
	{
		for(size_t j=0;j<n;j++){
			T[j]= fmodf(U[j],p);
			if (!positive)
				T[j]-=(T[j]>max)?p:0;
			T[j]+=(T[j]<min)?p:0;
		}

	}

#endif // AVX or AVX2

} // vectorised
} // FFLAS

// FADD
namespace FFLAS { namespace vectorised {

#if defined(__AVX2__) || defined(__AVX__)

#define VEC_ADD_D_POS(C,A,B,P,Q)				        \
	{								\
		C = _mm256_add_pd(A, B) ;                               \
		Q = _mm256_cmp_pd(C,P,_CMP_GE_OS);			\
		Q = _mm256_and_pd(Q,P);				        \
		C = _mm256_sub_pd(C,Q);					\
	}

#define VEC_ADD_D(C,A,B,P,NEGP,T,MIN,MAX)				\
	{								\
		C = _mm256_add_pd(A, B) ;                               \
		Q = _mm256_cmp_pd(C,MAX,_CMP_GT_OS);			\
		T = _mm256_cmp_pd(C,MIN,_CMP_LT_OS);			\
		Q = _mm256_and_pd(Q,NEGP);				\
		T = _mm256_and_pd(T,P);					\
		Q = _mm256_or_pd(Q,T);				        \
		C = _mm256_add_pd(C,Q);					\
	}

#define VEC_ADD_S_POS(C,A,B,P,Q)				        \
	{								\
		C = _mm256_add_ps(A, B) ;                               \
		Q = _mm256_cmp_ps(C,P,_CMP_GE_OS);			\
		Q = _mm256_and_ps(Q,P);				        \
		C = _mm256_sub_ps(C,Q);					\
	}

#define VEC_ADD_S(C,A,B,P,NEGP,T,MIN,MAX)				\
	{								\
		C = _mm256_add_ps(A, B) ;                               \
		Q = _mm256_cmp_ps(C,MAX,_CMP_GT_OS);			\
		T = _mm256_cmp_ps(C,MIN,_CMP_LT_OS);			\
		Q = _mm256_and_ps(Q,NEGP);				\
		T = _mm256_and_ps(T,P);					\
		Q = _mm256_or_ps(Q,T);				        \
		C = _mm256_add_ps(C,Q);					\
	}

	template<bool positive> // no default argument unless C++11
	inline void addp( double *T, const double * TA, const double * TB, size_t n, double p, double min, double max)
	{
		register __m256d A,B,C,Q,P,NEGP,TMP,MIN,MAX;
		P   = _mm256_set1_pd(p);
		NEGP= _mm256_set1_pd(-p);
		MIN = _mm256_set1_pd(min);
		MAX = _mm256_set1_pd(max);
		long st=long(T)%32;
		size_t i=0;
		if (n < 4) {
			for (;i<n;i++){
				T[i]=TA[i] + TB[i];
				T[i]-=(T[i]>max)?p:0;
				if (!positive)
					T[i]+=(T[i]<min)?p:0;
			}
			return;

		}
		if (st){ // the array T is not 32 byte aligned (process few elements s.t. (T+i) is 32 bytes aligned)
			for (size_t j=(size_t)st;j<32;j+=8,i++){
				T[i]=TA[i] + TB[i];
				T[i]-=(T[i]>max)?p:0;
				if (!positive)
					T[i]+=(T[i]<min)?p:0;
			}
		}
		FFLASFFPACK_check((long(T+i)%32==0));
		if ( (long(TA+i)%32==0) && (long(TB+i)%32==0)) {
			// perform the loop using 256 bits SIMD
			for (;i<=n-4;i+=4){
				C=_mm256_load_pd(T+i);
				A=_mm256_load_pd(TA+i);
				B=_mm256_load_pd(TB+i);
				if (positive) {
					VEC_ADD_D_POS(C,A,B,P,Q);
				}
				else {
					VEC_ADD_D(C,A,B,P,NEGP,TMP,MIN,MAX);
				}
				_mm256_store_pd(T+i,C);
			}
		}
		else {
#ifndef NDEBUG
			// std::cout << "in vectorised::addp, could not align T, Ta and Tb" << std::endl;
#endif
		}
		// perform the last elt from T without SIMD
		for (;i<n;i++){
			T[i]=TA[i] + TB[i];
			T[i]-=(T[i]>max)?p:0;
			if (!positive)
				T[i]+=(T[i]<min)?p:0;
		}
	}

	template<bool positive> // no default argument unless C++11
	inline void addp( float *T, const float * TA, const float * TB, size_t n, float p, float min, float max)
	{
		register __m256 A,B,C,Q,P,NEGP,TMP,MIN,MAX;
		P   = _mm256_set1_ps(p);
		NEGP= _mm256_set1_ps(-p);
		MIN = _mm256_set1_ps(min);
		MAX = _mm256_set1_ps(max);
		long st=long(T)%32;
		size_t i=0;
		if (n < 8) {
			for (;i<n;i++){
				T[i]=TA[i] + TB[i];
				T[i]-=(T[i]>max)?p:0;
				if (!positive)
				       	T[i]+=(T[i]<min)?p:0;
			}
			return;

		}
		if (st){ // the array T is not 32 byte aligned (process few elements s.t. (T+i) is 32 bytes aligned)
			for (size_t j=(size_t)st;j<32;j+=4,i++){
				T[i]=TA[i] + TB[i];
				T[i]-=(T[i]>max)?p:0;
				if (!positive)
					T[i]+=(T[i]<min)?p:0;
			}
		}
		FFLASFFPACK_check((long(T+i)%32==0));
		if ( (long(TA+i)%32==0) && (long(TB+i)%32==0)) {
			// perform the loop using 256 bits SIMD
			for (;i<=n-8;i+=8){
				C=_mm256_load_ps(T+i);
				A=_mm256_load_ps(TA+i);
				B=_mm256_load_ps(TB+i);
				if (positive) {
					VEC_ADD_S_POS(C,A,B,P,Q);
				}
				else {
					VEC_ADD_S(C,A,B,P,NEGP,TMP,MIN,MAX);
				}
				_mm256_store_ps(T+i,C);
			}
		}
		else {
#ifndef NDEBUG
			// std::cout << "in vectorised::addp, could not align T, Ta and Tb" << std::endl;
#endif
		}
		// perform the last elt from T without SIMD
		for (;i<n;i++){
			T[i]=TA[i] + TB[i];
			T[i]-=(T[i]>max)?p:0;
			if (!positive)
				T[i]+=(T[i]<min)?p:0;
		}
	}

#if 0
	inline void addp( float *T, const float * TA, const float * TB, size_t n)
	{
		register __m256 A,B,C;
		long st=long(T)%32;
		size_t i=0;
		if (n < 8) {
			for (;i<n;i++){
				T[i]=TA[i] + TB[i];
			}
			return;

		}
		if (st){ // the array T is not 32 byte aligned (process few elements s.t. (T+i) is 32 bytes aligned)
			for (size_t j=(size_t)st;j<32;j+=4,i++){
				T[i]=TA[i] + TB[i];
			}
		}
		FFLASFFPACK_check((long(T+i)%32==0));
		if ( (long(TA+i)%32==0) && (long(TB+i)%32==0)) {
			// perform the loop using 256 bits SIMD
			for (;i<=n-8;i+=8){
				C=_mm256_load_ps(T+i);
				A=_mm256_load_ps(TA+i);
				B=_mm256_load_ps(TB+i);
				C = _mm256_add_ps(A, B) ;
				_mm256_store_ps(T+i,C);
			}
		}
		else {
#ifndef NDEBUG
			std::cout << "in vectorised::addp, could not align T, Ta and Tb" << std::endl;
#endif
		}
		// perform the last elt from T without SIMD
		for (;i<n;i++){
			T[i]=TA[i] + TB[i];
		}
	}
#endif

#else // no AVX

	template<bool positive> // no default argument unless C++11
	inline void addp( double *T, const double * TA, const double * TB, size_t n, double p, double min, double max)
	{
		size_t i = 0 ;
		for (;i<n;i++){
			T[i]=TA[i] + TB[i];
			T[i]-=(T[i]>max)?p:0;
			if (!positive)
				T[i]+=(T[i]<min)?p:0;
		}

	}

	template<bool positive> // no default argument unless C++11
	inline void addp( float *T, const float * TA, const float * TB, size_t n, float p, float min, float max)
	{
		size_t i = 0 ;
		for (;i<n;i++){
			T[i]=TA[i] + TB[i];
			T[i]-=(T[i]>max)?p:0;
			if (!positive)
				T[i]+=(T[i]<min)?p:0;
		}

	}



#endif // AVX2 or AVX

} // vectorised
} // FFLAS

// FSUB
namespace FFLAS { namespace vectorised {

#if defined(__AVX2__) || defined(__AVX__)

#define VEC_SUB_D_POS(C,A,B,P,Q,MIN)				        \
	{								\
		C = _mm256_sub_pd(A, B) ;                               \
		Q = _mm256_cmp_pd(C,MIN,_CMP_LT_OS);			\
		Q = _mm256_and_pd(Q,P);				        \
		C = _mm256_add_pd(C,Q);					\
	}

#define VEC_SUB_D(C,A,B,P,NEGP,T,MIN,MAX)				\
	{								\
		C = _mm256_sub_pd(A, B) ;                               \
		Q = _mm256_cmp_pd(C,MAX,_CMP_GT_OS);			\
		T = _mm256_cmp_pd(C,MIN,_CMP_LT_OS);			\
		Q = _mm256_and_pd(Q,NEGP);				\
		T = _mm256_and_pd(T,P);					\
		Q = _mm256_or_pd(Q,T);				        \
		C = _mm256_add_pd(C,Q);					\
	}

#define VEC_SUB_S_POS(C,A,B,P,Q,MIN)				        \
	{								\
		C = _mm256_sub_ps(A, B) ;                               \
		Q = _mm256_cmp_ps(C,MIN,_CMP_GT_OS);			\
		Q = _mm256_and_ps(Q,P);				        \
		C = _mm256_add_ps(C,Q);					\
	}

#define VEC_SUB_S(C,A,B,P,NEGP,T,MIN,MAX)				\
	{								\
		C = _mm256_sub_ps(A, B) ;                               \
		Q = _mm256_cmp_ps(C,MAX,_CMP_GT_OS);			\
		T = _mm256_cmp_ps(C,MIN,_CMP_LT_OS);			\
		Q = _mm256_and_ps(Q,NEGP);				\
		T = _mm256_and_ps(T,P);					\
		Q = _mm256_or_ps(Q,T);				        \
		C = _mm256_add_ps(C,Q);					\
	}

	template<bool positive> // no default argument unless C++11
	inline void subp( double *T, const double * TA, const double * TB, size_t n, double p, double min, double max)
	{
		register __m256d A,B,C,Q,P,NEGP,TMP,MIN,MAX;
		P   = _mm256_set1_pd(p);
		NEGP= _mm256_set1_pd(-p);
		MIN = _mm256_set1_pd(min);
		MAX = _mm256_set1_pd(max);
		long st=long(T)%32;
		size_t i=0;
		if (n < 4) {
			for (;i<n;i++){
				T[i]=TA[i] - TB[i];
				if (!positive)
					T[i]-=(T[i]>max)?p:0;
				T[i]+=(T[i]<min)?p:0;
			}
			return;

		}
		if (st){ // the array T is not 32 byte aligned (process few elements s.t. (T+i) is 32 bytes aligned)
			for (size_t j=(size_t)st;j<32;j+=8,i++){
				T[i]=TA[i] - TB[i];
				if (!positive)
					T[i]-=(T[i]>max)?p:0;
				T[i]+=(T[i]<min)?p:0;
			}
		}
		FFLASFFPACK_check((long(T+i)%32==0));
		if ( (long(TA+i)%32==0) && (long(TB+i)%32==0)) {
			// perform the loop using 256 bits SIMD
			for (;i<=n-4;i+=4){
				C=_mm256_load_pd(T+i);
				A=_mm256_load_pd(TA+i);
				B=_mm256_load_pd(TB+i);
				if (positive) {
					VEC_SUB_D_POS(C,A,B,P,Q,MIN);
				}
				else {
					VEC_SUB_D(C,A,B,P,NEGP,TMP,MIN,MAX);
				}
				_mm256_store_pd(T+i,C);
			}
		}
		else {
#ifndef NDEBUG
			// std::cout << "in vectorised::subp, could not align T, Ta and Tb" << std::endl;
#endif
		}
		// perform the last elt from T without SIMD
		for (;i<n;i++){
			T[i]=TA[i] - TB[i];
			if (!positive)
				T[i]-=(T[i]>max)?p:0;
			T[i]+=(T[i]<min)?p:0;
		}
	}

	template<bool positive> // no default argument unless C++11
	inline void subp( float *T, const float * TA, const float * TB, size_t n, float p, float min, float max)
	{
		register __m256 A,B,C,Q,P,NEGP,TMP,MIN,MAX;
		P   = _mm256_set1_ps(p);
		NEGP= _mm256_set1_ps(-p);
		MIN = _mm256_set1_ps(min);
		MAX = _mm256_set1_ps(max);
		long st=long(T)%32;
		size_t i=0;
		if (n < 8) {
			for (;i<n;i++){
				T[i]=TA[i] - TB[i];
				if (!positive)
					T[i]-=(T[i]>max)?p:0;
				T[i]+=(T[i]<min)?p:0;
			}
			return;

		}
		if (st){ // the array T is not 32 byte aligned (process few elements s.t. (T+i) is 32 bytes aligned)
			for (size_t j=(size_t)st;j<32;j+=4,i++){
				T[i]=TA[i] - TB[i];
				if (!positive)
					T[i]-=(T[i]>max)?p:0;
				T[i]+=(T[i]<min)?p:0;
			}
		}
		FFLASFFPACK_check((long(T+i)%32==0));
		if ( (long(TA+i)%32==0) && (long(TB+i)%32==0)) {
			// perform the loop using 256 bits SIMD
			for (;i<=n-8;i+=8){
				C=_mm256_load_ps(T+i);
				A=_mm256_load_ps(TA+i);
				B=_mm256_load_ps(TB+i);
				if (positive) {
					VEC_SUB_S_POS(C,A,B,P,Q,MIN);
				}
				else {
					VEC_SUB_S(C,A,B,P,NEGP,TMP,MIN,MAX);
				}
				_mm256_store_ps(T+i,C);
			}
		}
		else {
#ifndef NDEBUG
			// std::cout << "in vectorised::subp, could not align T, Ta and Tb" << std::endl;
#endif
		}
		// perform the last elt from T without SIMD
		for (;i<n;i++){
			T[i]=TA[i] - TB[i];
			if (!positive)
				T[i]-=(T[i]>max)?p:0;
			T[i]+=(T[i]<min)?p:0;
		}
	}

#if 0
	inline void subp( float *T, const float * TA, const float * TB, size_t n)
	{
		register __m256 A,B,C;
		long st=long(T)%32;
		size_t i=0;
		if (n < 8) {
			for (;i<n;i++){
				T[i]=TA[i] - TB[i];
			}
			return;

		}
		if (st){ // the array T is not 32 byte aligned (process few elements s.t. (T+i) is 32 bytes aligned)
			for (size_t j=(size_t)st;j<32;j+=4,i++){
				T[i]=TA[i] - TB[i];
			}
		}
		FFLASFFPACK_check((long(T+i)%32==0));
		if ( (long(TA+i)%32==0) && (long(TB+i)%32==0)) {
			// perform the loop using 256 bits SIMD
			for (;i<=n-8;i+=8){
				C=_mm256_load_ps(T+i);
				A=_mm256_load_ps(TA+i);
				B=_mm256_load_ps(TB+i);
				C = _mm256_sub_ps(A, B) ;
				_mm256_store_ps(T+i,C);
			}
		}
		else {
#ifndef NDEBUG
			// std::cout << "in vectorised::subp, could not align T, Ta and Tb" << std::endl;
#endif
		}
		// perform the last elt from T without SIMD
		for (;i<n;i++){
			T[i]=TA[i] - TB[i];
		}
	}
#endif

#else // no AVX

	template<bool positive> // no default argument unless C++11
	inline void subp( double *T, const double * TA, const double * TB, size_t n, double p, double min, double max)
	{
		size_t i = 0 ;
		for (;i<n;i++){
			T[i]=TA[i] - TB[i];
			if (!positive)
				T[i]-=(T[i]>max)?p:0;
			T[i]+=(T[i]<min)?p:0;
		}

	}

	template<bool positive> // no default argument unless C++11
	inline void subp( float *T, const float * TA, const float * TB, size_t n, float p, float min, float max)
	{
		size_t i = 0 ;
		for (;i<n;i++){
			T[i]=TA[i] - TB[i];
			if (!positive)
				T[i]-=(T[i]>max)?p:0;
			T[i]+=(T[i]<min)?p:0;
		}

	}



#endif // AVX2 or AVX

} // vectorised
} // FFLAS

// FAXPY

// FSCAL

#endif // __FFLASFFPACK_fflas_avx_functions_H