void uncertainty_pnp(
    double* pts2d,  // pn,2
    double* pts3d,  // pn,3
    double* wgt2d,  // pn,3 wxx,wxy,wyy
    double* K,      // 3,3
    double* init_rt,// 6
    double* result_rt,// 6
    int pn
);
