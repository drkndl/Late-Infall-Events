#include "fargo3d.h"
#include <stdlib.h>
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define M_P 1.672621911e-24
#define N_A 6.022e23
#define AU 1.49597871e13
#define IN_DISK 0
#define IN_CLOUDLET 1
#define CUTOFF_WIDTH_IN (RIN*0.05)
#define CUTOFF_WIDTH_OUT (ROUT*0.05)
#define B_CRIT (G*MSTAR/VINF/VINF)
#define INT_GRID_SIZE 100
#define K_MIN (M_PI/CLOUDLETRADIUS)
#define RAND_UNIFORM() ((real)rand()/RAND_MAX)
typedef struct vector3 Vector3;
typedef struct vector2 Vector2;

struct vector3 {
  real x;
  real y;
  real z;
};
struct vector2 {
  real x;
  real y;
};
struct cloudlet_orbit {
  Vector3 pos;
  Vector3 vel;
};
struct cloudlet_orbit CloudletInit();
Vector3 Rotate(Vector3 vec, real rotangle_x, real rotangle_y, boolean rotaxfirst, boolean prograde);

#ifdef ITURBULENCE
real rand_gauss(real mean, real std);
struct turb_field {
  Vector3* kwav;
  Vector3* delta_v0;
  real* phases;
};
struct turb_field InitTurbField() {
  struct turb_field turbulenceField;
  real sum_v0_2 = 0.0;
  turbulenceField.kwav = (Vector3*)malloc(NWAVES*sizeof(Vector3));
  turbulenceField.delta_v0 = (Vector3*)malloc(NWAVES*sizeof(Vector3));
  turbulenceField.phases = (real*)malloc(NWAVES*sizeof(real));
  for (int n = 0; n < NWAVES; n++) {
    real k_abs = pow(1.0/(1.0-RAND_UNIFORM()),(1.0/(PSINDEX-1.0)))*K_MIN;
    real deltav_abs = rand_gauss(0.0, sqrt(pow(k_abs, -PSINDEX)));
    real k_phi = 2.0*M_PI*RAND_UNIFORM();
    real k_theta = acos(2.0*RAND_UNIFORM()-1);
    real deltav_phi = 2.0*M_PI*RAND_UNIFORM();
    real deltav_theta = atan(-1.0/(tan(k_theta)*(2.0*sin(k_phi)*sin(deltav_phi)+cos(k_phi+deltav_phi))));
    turbulenceField.kwav[n].x = k_abs*sin(k_theta)*cos(k_phi);
    turbulenceField.kwav[n].y = k_abs*sin(k_theta)*sin(k_phi);
    turbulenceField.kwav[n].z = k_abs*cos(k_theta);
    turbulenceField.delta_v0[n].x = deltav_abs*sin(k_theta)*cos(k_phi);
    turbulenceField.delta_v0[n].y = deltav_abs*sin(k_theta)*sin(k_phi);
    turbulenceField.delta_v0[n].z = deltav_abs*cos(k_theta);
    turbulenceField.phases[n] = 2.0*M_PI*RAND_UNIFORM();
    sum_v0_2 = sum_v0_2 + deltav_abs*deltav_abs;
  }
  real norm = sqrt(2*G*CLOUDLETMASS/CLOUDLETRADIUS/sum_v0_2);
  for (int n = 0; n < NWAVES; n++) {
    turbulenceField.delta_v0[n].x = turbulenceField.delta_v0[n].x * norm * TURBFACTOR;
    turbulenceField.delta_v0[n].y = turbulenceField.delta_v0[n].y * norm * TURBFACTOR;
    turbulenceField.delta_v0[n].z = turbulenceField.delta_v0[n].z * norm * TURBFACTOR;
  }
  return turbulenceField;
}

Vector3 TurbVel(real xcar, real ycar, real zcar, struct turb_field field) {
  Vector3 vel = {0.0, 0.0, 0.0};
  for (int n = 0; n < NWAVES; n++) {
    vel.x = vel.x + field.delta_v0[n].x * cos(field.kwav[n].x * xcar + field.kwav[n].y * ycar + field.kwav[n].z * zcar + field.phases[n]);
    vel.y = vel.y + field.delta_v0[n].y * cos(field.kwav[n].x * xcar + field.kwav[n].y * ycar + field.kwav[n].z * zcar + field.phases[n]);
    vel.z = vel.z + field.delta_v0[n].z * cos(field.kwav[n].x * xcar + field.kwav[n].y * ycar + field.kwav[n].z * zcar + field.phases[n]);
  }
  return vel;
}
#endif
#ifdef ICLOUDLET
real GetTrueanomaly(real r, real b_bcrit, char sign) {
    return sign*acos((b_bcrit*b_bcrit*B_CRIT-r)/(r*sqrt(1+b_bcrit*b_bcrit)));
}

real GetVelocityAngle(real trueanomaly, real b_bcrit) {
    real flightpathangle = atan(sin(trueanomaly)/(cos(trueanomaly)+1/sqrt(1+b_bcrit*b_bcrit)));
    return flightpathangle - trueanomaly + M_PI/2;
}
    
Vector3 Rotate(Vector3 vec, real rotangle_x, real rotangle_y, boolean rotaxfirst, boolean prograde) {
  Vector3 res;
  real _rotangle_x = rotangle_x * M_PI / 180.0;
  real _rotangle_y = rotangle_y * M_PI / 180.0;
  if (prograde) {
    vec.y = -vec.y;
  }
  res.x = vec.x;
  res.y = vec.y;
  res.z = vec.z;
  if (rotangle_x != 0.0 && rotangle_y == 0.0) {
    // rotation only around x-axis
    res.y = cos(_rotangle_x) * vec.y - sin(_rotangle_x) * vec.z;
    res.z = sin(_rotangle_x) * vec.y + cos(_rotangle_x) * vec.z;
  }
  else if (rotangle_y != 0.0 && rotangle_x == 0.0) {
    // rotation only around y-axis
    res.x = cos(_rotangle_y) * vec.x + sin(_rotangle_y) * vec.z;
    res.z = -sin(_rotangle_y) * vec.x + cos(_rotangle_y) * vec.z;
  }
  else if (rotangle_x !=0.0 && rotangle_y != 0.0 && !rotaxfirst){
    // first rotation around x axis
    real y = cos(_rotangle_x) * vec.y - sin(_rotangle_x) * vec.z;
    real z = sin(_rotangle_x) * vec.y + cos(_rotangle_x) * vec.z;
    // then rotation around y axis
    res.x = cos(_rotangle_y) * vec.x + sin(_rotangle_y) * z;
    res.y = y;
    res.z = -sin(_rotangle_y) * vec.x + cos(_rotangle_y) * z;
  }
  else if (rotangle_x != 0.0 && rotangle_y != 0.0 && rotaxfirst){
    // first rotation around y axis
    real x = cos(_rotangle_y) * vec.x + sin(_rotangle_y) * vec.z;
    real z = -sin(_rotangle_y) * vec.x + cos(_rotangle_y) * vec.z;
    // then rotation around x axis
    res.x = x;
    res.y = cos(_rotangle_x) * vec.y - sin(_rotangle_x) * z;
    res.z = sin(_rotangle_x) * vec.y + cos(_rotangle_x) * z;
  }
  return res;
}

struct cloudlet_orbit GetOrbit(real dist_cloud, real b_bcrit, char sign) {
    real trueanomaly = GetTrueanomaly(dist_cloud, b_bcrit, sign);
    real speed = VINF * sqrt(2*B_CRIT/dist_cloud+1);
    real velocityangle = GetVelocityAngle(trueanomaly, b_bcrit);
    real x_ini = cos(trueanomaly)*dist_cloud;
    real y_ini = sin(trueanomaly)*dist_cloud;
    real z_ini = 0.0;
    real vx_ini = speed * cos(velocityangle);
    real vy_ini = -speed * sin(velocityangle);
    real vz_ini = 0.0;
    Vector3 pos = {x_ini, y_ini, z_ini};
    Vector3 vel = {vx_ini, vy_ini, vz_ini};
    struct cloudlet_orbit orbit = {pos, vel};
    return orbit;
}

real Sum(Vector3 a) {
  return a.x+a.y+a.z;
}

Vector3 CrossProduct(Vector3 a, Vector3 b) {
  Vector3 res = {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
  return res;
}

Vector3 ElementProduct(Vector3 a, Vector3 b) {
  Vector3 res = {a.x*b.x, a.y*b.y, a.z*b.z};
  return res;
}

real DotProduct(Vector3 a, Vector3 b) {
  return Sum(ElementProduct(a,b));
}

Vector3 ElementSum(Vector3 a, Vector3 b) {
  Vector3 res = {a.x+b.x, a.y+b.y, a.z+b.z};
  return res;
}

Vector3 ScalarDivide(Vector3 a, real b) {
  Vector3 res = {a.x/b, a.y/b, a.z/b};
  return res;
}

struct basis {
  Vector3 e1;
  Vector3 e2;
  Vector3 e3;
};
struct basis PlaneBasis(Vector3 rvec, Vector3 vvec) {
  Vector3 n = CrossProduct(vvec, rvec);
  Vector3 a1 = {1, 0, -n.x/n.z};
  Vector3 a2 = CrossProduct(n, a1);
  struct basis res = {
    ScalarDivide(a1,sqrt(DotProduct(a1,a1))),
    ScalarDivide(a2,sqrt(DotProduct(a2,a2))),
    ScalarDivide(n,sqrt(DotProduct(n,n)))
  };
  return res;
}

Vector3 TransformToPlane(Vector3 vec, Vector3 e1, Vector3 e2, Vector3 e3) {
  Vector3 res = {DotProduct(vec, e1), DotProduct(vec, e2), DotProduct(vec, e3)};
  return res;
}

Vector3 TransformToOriginal(Vector3 vec_prime, Vector3 e1, Vector3 e2, Vector3 e3) {
  Vector3 res = {
    vec_prime.x * e1.x + vec_prime.y * e2.x + vec_prime.z * e3.x,
    vec_prime.x * e1.y + vec_prime.y * e2.y + vec_prime.z * e3.y,
    vec_prime.x * e1.z + vec_prime.y * e2.z + vec_prime.z * e3.z
  };
  return res;
}

char Sign(real a) {
  if (a >= 0) {
    return 1;
  } else {
    return -1;
  }
}

Vector2 OrbitToStraight2D(real x_orbit, real y_orbit, real b_bcrit_orbit) {
  real th = Sign(b_bcrit_orbit)*atan(fabs(b_bcrit_orbit));
  Vector2 res = {x_orbit*cos(th)-y_orbit*sin(th), x_orbit*sin(th)+y_orbit*cos(th)};
  return res;
}

Vector2 StraightToOrbit2D(real x_straight, real y_straight, real b_bcrit_orbit) {
  real th = -Sign(b_bcrit_orbit)*atan(fabs(b_bcrit_orbit));
  Vector2 res = {x_straight*cos(th)-y_straight*sin(th), x_straight*sin(th)+y_straight*cos(th)};
  return res;
}

real Get_e(real b_bcrit) {
  return sqrt(1+b_bcrit*b_bcrit);
}

real Get_r(real H, real b_bcrit) {
  real e = Get_e(b_bcrit);
  return -B_CRIT*(1-e*cosh(H));
}

Vector2 GetPosition2D(real H, real b_bcrit) {
  real e = Get_e(b_bcrit);
  real trueanomaly = -2*atan(sqrt((e+1)/(e-1))*tanh(H/2));
  real r = Get_r(H, b_bcrit);
  return OrbitToStraight2D(r*cos(trueanomaly), r*sin(trueanomaly), b_bcrit);
}

Vector2 GetVelocity2D(real H, real b_bcrit) {
  real e = Get_e(b_bcrit);
  real trueanomaly = -2*atan(sqrt((e+1)/(e-1))*tanh(H/2));
  real velocityangle = GetVelocityAngle(trueanomaly, b_bcrit);
  real r = Get_r(H, b_bcrit);
  real speed = VINF * sqrt(2*B_CRIT/r+1);
  return OrbitToStraight2D(speed * cos(velocityangle), -speed * sin(velocityangle), b_bcrit);
}

real GetHFromTrueanomaly(real trueanomaly, real e) {
  return 2*atanh(sqrt((e-1)/(e+1))*tan(-trueanomaly/2));
}

real Halley_f(real M, real H, real e) {
  return -e*sinh(H)+H+M;
}

real Halley_fprime(real H, real e) {
  return -e*cosh(H)+1;
}

real Halley_fprime2(real H, real e) {
  return -e*sinh(H);
}

real Halley(real M, real e) {
  real prev_root = M;
  real root = M;
  for (int n = 0; n < 50; n++) {
    real f = Halley_f(M, prev_root, e);
    if (f == 0) {
      return prev_root;
    }
    real f_prime = Halley_fprime(prev_root, e);
    real f_prime2 = Halley_fprime2(prev_root, e);
    real newton_step = f/f_prime;
    real adj = newton_step * f_prime2 / f_prime / 2;
    if (fabs(adj) < 1) {
      newton_step = newton_step / (1-adj);
    }
    root = prev_root - newton_step;
    if (fabs(root-prev_root) < 1.48e-8) {
      break;
    }
    prev_root = root;
  }
  return root;
}

real Get_H(real t, real b_bcrit) {
  real M = VINF/B_CRIT * t;
  real e = Get_e(b_bcrit);
  return Halley(M, e);
}

real Get_M(real trueanomaly, real b_bcrit) {
  real e = Get_e(b_bcrit);
  real H = GetHFromTrueanomaly(trueanomaly, e);
  return e*sinh(H)-H;
}

real Get_t(real r, real b_bcrit, char sign) {
  real M = Get_M(GetTrueanomaly(r, b_bcrit, sign), b_bcrit);
  return M * B_CRIT/VINF;
}

struct hyperbolic_cloudlet {
  Vector3* pos;
  Vector3* vel;
  int ncloudlet;
};
struct hyperbolic_cloudlet GetCloudletPoints() {
  real* x = malloc(NHYPERBOLIC*sizeof(real));
  real* y = malloc(NHYPERBOLIC*sizeof(real));
  real* z = malloc(NHYPERBOLIC*sizeof(real));
  for (int n = 0; n < NHYPERBOLIC; n++) {
    x[n] = y[n] = z[n] = -CLOUDLETRADIUS + 2 * n / (real)(NHYPERBOLIC-1) * CLOUDLETRADIUS;
  }
  struct cloudlet_orbit oini = GetOrbit(RC0, fabs(IMPACTPARAMETER), Sign(IMPACTPARAMETER));
  Vector3 ini = {oini.pos.x, oini.pos.y, oini.pos.z};
  Vector2 _ini2d = OrbitToStraight2D(ini.x, ini.y, IMPACTPARAMETER);
  ini.x = _ini2d.x;
  ini.y = _ini2d.y;
  real tau0 = Get_t(RC0, IMPACTPARAMETER, Sign(IMPACTPARAMETER));
  real tau1 = Get_t(DISTINI, IMPACTPARAMETER, Sign(IMPACTPARAMETER));
  real tau = tau1-tau0;
  real* xcl = malloc(NHYPERBOLIC*NHYPERBOLIC*NHYPERBOLIC*sizeof(real));
  real* ycl = malloc(NHYPERBOLIC*NHYPERBOLIC*NHYPERBOLIC*sizeof(real));
  real* zcl = malloc(NHYPERBOLIC*NHYPERBOLIC*NHYPERBOLIC*sizeof(real));
  real* vxcl = malloc(NHYPERBOLIC*NHYPERBOLIC*NHYPERBOLIC*sizeof(real));
  real* vycl = malloc(NHYPERBOLIC*NHYPERBOLIC*NHYPERBOLIC*sizeof(real));
  real* vzcl = malloc(NHYPERBOLIC*NHYPERBOLIC*NHYPERBOLIC*sizeof(real));
  int ncloudlet = 0;
  struct hyperbolic_cloudlet res;
  for (int iii = 0; iii < NHYPERBOLIC; iii++) {
    for (int jjj = 0; jjj < NHYPERBOLIC; jjj++) {
      for (int kkk = 0; kkk < NHYPERBOLIC; kkk++) {
        if (x[iii]*x[iii]+y[jjj]*y[jjj]+z[kkk]*z[kkk] <= CLOUDLETRADIUS*CLOUDLETRADIUS) {
          Vector3 curr = {x[iii], y[jjj], z[kkk]};
          Vector3 rvec = ElementSum(ini, curr);
          Vector3 vvec = {VINF, 0, 0};
          struct basis pbasis = PlaneBasis(rvec, vvec);
          Vector3 rvec_prime = TransformToPlane(rvec, pbasis.e1, pbasis.e2, pbasis.e3);
          real b = rvec_prime.y/B_CRIT;
          real r = sqrt(DotProduct(rvec_prime, rvec_prime));
          char sign = Sign(b);
          b = fabs(b);
          real t0 = Get_t(r, b, sign);
          real t;
          if (sign < 0) {
              t = t0-fabs(tau);
          } else {
              t = t0+fabs(tau);
          }
          real H = Get_H(t, b);
          Vector3 pos_prime;
          Vector2 pos2d_prime = GetPosition2D(H, sign*b);
          Vector3 vel_prime;
          Vector2 vel2d_prime = GetVelocity2D(H, sign*b);
          pos_prime.x = pos2d_prime.x;
          pos_prime.y = pos2d_prime.y;
          pos_prime.z = 0.0;
          vel_prime.x = vel2d_prime.x;
          vel_prime.y = vel2d_prime.y;
          vel_prime.z = 0.0;
          Vector3 pos_final = TransformToOriginal(pos_prime, pbasis.e1, pbasis.e2, pbasis.e3);
          Vector3 vel_final = TransformToOriginal(vel_prime, pbasis.e1, pbasis.e2, pbasis.e3);
          Vector2 pos_straight2d;
          pos_straight2d.x = pos_final.x;
          pos_straight2d.y = pos_final.y;
          Vector2 pos_orbit2d = StraightToOrbit2D(pos_straight2d.x, pos_straight2d.y, IMPACTPARAMETER);
          pos_final.x = pos_orbit2d.x;
          pos_final.y = pos_orbit2d.y;
          Vector2 vel_straight2d;
          vel_straight2d.x = vel_final.x;
          vel_straight2d.y = vel_final.y;
          Vector2 vel_orbit2d = StraightToOrbit2D(vel_straight2d.x, vel_straight2d.y, IMPACTPARAMETER);
          vel_final.x = vel_orbit2d.x;
          vel_final.y = vel_orbit2d.y;
          xcl[ncloudlet] = pos_final.x;
          ycl[ncloudlet] = pos_final.y;
          zcl[ncloudlet] = pos_final.z;
          vxcl[ncloudlet] = vel_final.x;
          vycl[ncloudlet] = vel_final.y;
          vzcl[ncloudlet] = vel_final.z;
          ncloudlet++;
        }
      }
    }
  }
  res.pos = malloc(ncloudlet*sizeof(Vector3));
  res.vel = malloc(ncloudlet*sizeof(Vector3));
  res.ncloudlet = ncloudlet;
  for (int n = 0; n < ncloudlet; n++) {
    Vector3 pos = {xcl[n], ycl[n], zcl[n]};
    Vector3 vel = {vxcl[n], vycl[n], vzcl[n]};
    Vector3 rotated_pos = Rotate(pos, ROTANGLEX, ROTANGLEY, ROTAXFIRST, PROGRADE);
    Vector3 rotated_vel = Rotate(vel, ROTANGLEX, ROTANGLEY, ROTAXFIRST, PROGRADE);
    res.pos[n] = rotated_pos;
    res.vel[n] = rotated_vel;
  }
  free(x);
  free(y);
  free(z);
  free(xcl);
  free(ycl);
  free(zcl);
  free(vxcl);
  free(vycl);
  free(vzcl);
  return res;
}
#endif

void _Init(boolean is_tracer) {

#ifdef ITURBULENCE
  struct turb_field turbulenceField;
  if (!is_tracer) {
    srandom(TURBSEED);
    turbulenceField = InitTurbField();
  }
#endif

  int i,j,k;
  real *v1;
  real *v2;
  real *v3;
  real *e;
  real *rho;

  real omega;
  real r, r3, rcyl;
  real zcyl;

  real cs_iso, h;
  real z_o, z_i;

  rho = Density->field_cpu;
  e   = Energy->field_cpu;
  v1  = Vx->field_cpu;
  v2  = Vy->field_cpu;
  v3  = Vz->field_cpu;

#ifdef CYLINDRICAL
  fprintf(stderr, "ERROR: Cylindrical coordinates are currently not supported");
#endif

  real rho_cloudlet = CLOUDLETMASS*3/(4*M_PI*pow(CLOUDLETRADIUS,3.));

#if defined(ICLOUDLET) && ! defined(HYPERBOLIC)
  struct cloudlet_orbit orbit = CloudletInit();
#endif

#if defined(HYPERBOLIC) && defined(ICLOUDLET)
  struct hyperbolic_cloudlet cloudlet = GetCloudletPoints();
  real mass_per_particle = CLOUDLETMASS/cloudlet.ncloudlet;
#endif

  for (k=0; k<Nz+2*NGHZ; k++) {
    for (j=0; j<Ny+2*NGHY; j++) {

      r = Ymed(j);
      r3 = r*r*r;
      rcyl = r * sin(Zmed(k));
      zcyl = r * cos(Zmed(k));
      omega = sqrt(G*MSTAR/(r3));
      h = ASPECTRATIO*pow(rcyl/R0,FLARINGINDEX);
      cs_iso = h*rcyl*omega;

      z_o = (rcyl-ROUT)/CUTOFF_WIDTH_OUT;
      z_i = (RIN-rcyl)/CUTOFF_WIDTH_IN;

      for (i=0; i<Nx; i++) {

        real component;
        // Convert current cell coordinates to cartesian coordinates
        real xcar = r*sin(Zmed(k))*cos(Xmed(i));
        real ycar = r*sin(Zmed(k))*sin(Xmed(i));
        real zcar = r*cos(Zmed(k));

#if defined(ICLOUDLET) && ! defined(HYPERBOLIC)
        // Distance to the cloudlet center
        real dist_cloudlet = sqrt((xcar-orbit.pos.x)*(xcar-orbit.pos.x)+(ycar-orbit.pos.y)*(ycar-orbit.pos.y)+(zcar-orbit.pos.z)*(zcar-orbit.pos.z));
        if (dist_cloudlet <= CLOUDLETRADIUS) {
          component = IN_CLOUDLET;
        } else {
          component = IN_DISK;
        }
#else
        component = IN_DISK;
#endif

        // Set the temperature / energy / sound speed
#ifdef ISOTHERMAL
        // Want the temperature to depend on rsph, rather than rcyl
        // Even though that shifts the disk out of thermal equilibrium at high z/r,
        // this ensures the correct temperature for out-of-plane cloudlet accretion,
        // and the disk will equilibrate during the simulation.
        real cs_sph = ASPECTRATIO*pow(r/R0, FLARINGINDEX)*omega*r;
        // Add floor temperature
        real T_sph = cs_sph*cs_sph*N_A*M_P/R_MU;
        e[l] = sqrt(R_MU/N_A/M_P*pow(T_sph*T_sph*T_sph*T_sph+TFLOOR*TFLOOR*TFLOOR*TFLOOR,0.25));
#else // ADIABATIC
        if (component != IN_CLOUDLET) {
          e[l] = cs_iso*cs_iso*rho[l]/(GAMMA-1.0);
        } else {
          e[l] = R_MU/N_A/M_P*TBG*rho[l]/(GAMMA-1.0);
        }
#endif
#ifdef IDISK
        if (!is_tracer) {
          rho[l] = SIGMA0/(sqrt(2.*M_PI)*ASPECTRATIO*R0)*pow(rcyl/R0, -SIGMASLOPE-FLARINGINDEX-1.)\
            *exp(-zcyl*zcyl/(2.*ASPECTRATIO*ASPECTRATIO*R0*R0)*pow(rcyl/R0,-2.*FLARINGINDEX-2.))\
            /(1+exp(z_o));
          if (RIN > 0) {
            rho[l] = rho[l] / (1+exp(z_i));
          }
        }
          real v1_kep_square = omega*omega*rcyl*rcyl;
          real v1_corr_square = cs_iso*cs_iso*(\
            (FLARINGINDEX+1.)*pow(zcyl/rcyl,2.)/(h*h)\
            -rcyl/(CUTOFF_WIDTH_OUT*(1.+exp(-z_o)))\
            -SIGMASLOPE+FLARINGINDEX-2.\
          );
          if (RIN > 0) {
            v1_corr_square = v1_corr_square + cs_iso*cs_iso*rcyl/(CUTOFF_WIDTH_IN*(1.+exp(-z_i)));
          }
          if (v1_corr_square + v1_kep_square < 0) {
            v1[l] = sqrt(v1_kep_square);
          } else {
            v1[l] = sqrt(v1_kep_square + v1_corr_square);
          }
          v2[l] = v3[l] = 0.0;
#else
          rho[l] = RHOFLOORGAS;
          v1[l] = v2[l] = v3[l] = 0.0;
#endif

#if defined(ICLOUDLET) && ! defined(HYPERBOLIC)
        if (component == IN_CLOUDLET) {
          rho[l] = rho_cloudlet;
          // Convert cartesian speeds that are returned by Lina's orbital math to spherical coordinates
          real vx = orbit.vel.x;
          real vy = orbit.vel.y;
          real vz = orbit.vel.z;
#ifdef ITURBULENCE
          if (!is_tracer) {
            Vector3 v_turb = TurbVel(xcar-orbit.pos.x, ycar-orbit.pos.y, zcar-orbit.pos.z, turbulenceField);
            vx = vx + v_turb.x;
            vy = vy + v_turb.y;
            vz = vy + v_turb.z;
          }
#endif
          v1[l] = vy*cos(Xmin(i)) - vx*sin(Xmin(i));
          v2[l] = vx*sin(Zmin(k))*cos(Xmin(i))+vy*sin(Zmin(k))*sin(Xmin(i))+vz*cos(Zmin(k));
          v3[l] = vx*cos(Zmin(k))*cos(Xmin(i))+vy*cos(Zmin(k))*sin(Xmin(i))-vz*sin(Zmin(k));
        }
#endif
        // Make sure that the floor density is enforced
        rho[l] = MAX(rho[l], RHOFLOORGAS);

      }
    }
  }
#if defined(ICLOUDLET) && defined(HYPERBOLIC)
  for (int n = 0; n < cloudlet.ncloudlet; n++) {
    real _phi = atan2(cloudlet.pos[n].y, cloudlet.pos[n].x);
    real _r = sqrt(cloudlet.pos[n].x*cloudlet.pos[n].x+cloudlet.pos[n].y*cloudlet.pos[n].y+cloudlet.pos[n].z*cloudlet.pos[n].z);
    real _theta = acos(cloudlet.pos[n].z/_r);
    int _i = (int)(NGHX+NX*(_phi-XMIN)/(XMAX-XMIN));
    int _j = (int)(NGHY-Y0+NY*(log(_r)-log(YMIN))/(log(YMAX)-log(YMIN)));
    int _k = (int)(NGHZ-Z0+NZ*(_theta-ZMIN)/(ZMAX-ZMIN));
    real len = MAX(MAX(edge_size_x(_i,_j,_k), edge_size_y(_j,_k)), edge_size_z(_j,_k));
    for (int ni = -2; ni < 3; ni++) {
      for (int nj = -2; nj < 3; nj++) {
        for (int nk = -2; nk < 3; nk++) {
            i = _i + ni;
            j = _j + nj;
            k = _k + nk;
            if (i < 0 || j < 0 || k < 0 || i >= Nx || j >= Ny+2*NGHY || k >= Nz+2*NGHZ) {
              // The cell belonging to this particle is on a different device
              continue;
            }
            real xcar = Ymed(j)*sin(Zmed(k))*cos(Xmed(i));
            real ycar = Ymed(j)*sin(Zmed(k))*sin(Xmed(i));
            real zcar = Ymed(j)*cos(Zmed(k));
            real vx = cloudlet.vel[n].x;
            real vy = cloudlet.vel[n].y;
            real vz = cloudlet.vel[n].z;
            real dist = sqrt((xcar-cloudlet.pos[n].x)*(xcar-cloudlet.pos[n].x)+(ycar-cloudlet.pos[n].y)*(ycar-cloudlet.pos[n].y)+(zcar-cloudlet.pos[n].z)*(zcar-cloudlet.pos[n].z));
            real q = dist/(2*len);
            if (q > 1) continue;
            real w3d;
            if (q <= 0.5) {
              w3d = 1-6*q*q+6*q*q*q;
            } else {
              w3d = 2*(1-q)*(1-q)*(1-q);
            }
            w3d = w3d * 8 / M_PI;
            real particle_dens = w3d*mass_per_particle*InvVol(i,j,k);
            v1[l] = (rho[l]*v1[l] + particle_dens*(vy*cos(Xmin(i)) - vx*sin(Xmin(i))))/(rho[l]+particle_dens);
            v2[l] = (rho[l]*v2[l] + particle_dens*(vx*sin(Zmin(k))*cos(Xmin(i))+vy*sin(Zmin(k))*sin(Xmin(i))+vz*cos(Zmin(k))))/(rho[l]+particle_dens);
            v3[l] = (rho[l]*v3[l] + particle_dens*(vx*cos(Zmin(k))*cos(Xmin(i))+vy*cos(Zmin(k))*sin(Xmin(i))-vz*sin(Zmin(k))))/(rho[l]+particle_dens);
            rho[l] = rho[l] + particle_dens;
          }
        }
      }
    }
#endif
#ifdef ITURBULENCE
  free(turbulenceField.kwav);
  free(turbulenceField.delta_v0);
  free(turbulenceField.phases);
#endif
#if defined(ICLOUDLET) && defined(HYPERBOLIC)
  free(cloudlet.pos);
  free(cloudlet.vel);
#endif
}

void Init() {
  _Init(FALSE);
}

void InitTracer() {
  _Init(TRUE);
}


void CondInit() {
  Fluids[0] = CreateFluid("gas",GAS);
  SelectFluid(0);
  Init();
  if (NFLUIDS > 1) {
    Fluids[1] = CreateFluid("tracer",DUST);
    SelectFluid(1);
    InitTracer();
    // Want St = 0, so that the dust can act as a tracer, so: 1/St -> Inf (first argument of ColRate)
    // NO -> No feedback between tracer dust and gas
    ColRate(1.0e10, 0, 1, NO);
  }
}

#if defined(ICLOUDLET) && ! defined(HYPERBOLIC)
struct cloudlet_orbit CloudletInit() {
  struct cloudlet_orbit orbit;
  orbit = GetOrbit(DISTINI, fabs(IMPACTPARAMETER), Sign(IMPACTPARAMETER));
  struct cloudlet_orbit rotated_orbit;
  rotated_orbit.pos = Rotate(orbit.pos, ROTANGLEX, ROTANGLEY, ROTAXFIRST, PROGRADE);
  rotated_orbit.vel = Rotate(orbit.vel, ROTANGLEX, ROTANGLEY, ROTAXFIRST, PROGRADE);
  return rotated_orbit;
}
#endif

#ifdef ITURBULENCE
real rand_gauss(real mean, real std) {
  return sqrt(-2.0*log(RAND_UNIFORM()))*cos(2.0*M_PI*RAND_UNIFORM())*std+mean;
}
#endif
