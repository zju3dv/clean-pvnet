void findNearestPointIdxLauncher(
    float* ref_pts,   // [b,pn1,dim]
    float* que_pts,   // [b,pn2,dim]
    int* idxs,        // [b,pn2]
    int b,
    int pn1,
    int pn2,
    int dim,
    int exclude_self
);
