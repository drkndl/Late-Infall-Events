//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void Floor_cpu() {

//<USER_DEFINED>
  INPUT(Density);
  OUTPUT(Density);
//<\USER_DEFINED>


//<EXTERNAL>
  real rhofloor_gas = RHOFLOORGAS;
  real rhofloor_dust = RHOFLOORDUST;
  real* dens = Density->field_cpu;
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
  int fluidtype = Fluidtype;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
//<\INTERNAL>

//<MAIN_LOOP>

  i = j = k = 0;

#ifdef Z
  for (k=0; k<size_z; k++) {
#endif
#ifdef Y
    for (j=0; j<size_y; j++) {
#endif
#ifdef X
      for (i=0; i<size_x; i++ ) {
#endif
//<#>
	ll = l;
  if (fluidtype == GAS && dens[ll]<rhofloor_gas)
    dens[ll] = rhofloor_gas;
  else if (fluidtype == DUST && dens[ll]<rhofloor_dust)
    dens[ll] = rhofloor_dust;
//<\#>
#ifdef X
      }
#endif
#ifdef Y
    }
#endif
#ifdef Z
  }
#endif
//<\MAIN_LOOP>
}
