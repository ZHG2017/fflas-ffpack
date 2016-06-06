function errCode = fadd_test_3(p)
  errCode = 0;
  try
    addpath(p);
    ff_rand_init();
    F = ff_init_Field();
    m = ff_init_Size();
    n = ff_init_Size();
    A = ff_init_Matrix(F, m, n);
    alpha = ff_init_Element(F);
    B = ff_init_Matrix(F, m, n);
    ref_res = modb(A+alpha*B,F);
    ff_res = fadd(F, A, alpha, B);
    eqTest = isequal(ff_res, ref_res);
    if ~eqTest
      error('Computation error');
    end
  catch exception;
    disp(exception.message);
    errCode = 1;
  end
end
