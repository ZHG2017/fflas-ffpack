% ftrsm right - lower - size(A) != size(B) B is not square matrix - alpha=1
function errCode = ftrsm_test_11(p)
  errCode = 0;
  try
    addpath(p);
    ff_rand_init();
    F = ff_init_Field();
    m = ff_init_Size();
    n = ff_init_Size();
    B = ff_init_Matrix(F,m,n);
    A = ff_init_Matrix(F,n,n);
    A = tril(A);
    ff_res = ftrsm(F, 'Right', 'Lower', 'NonUnit', 1, A, B);
    ref_res = modb( ff_res*A,F);

    eq = isequal(ref_res, B);
    if ~eq
      error('Computation error');
    end
  catch exception;
    disp(exception.message);
    errCode = 1;
  end
end
