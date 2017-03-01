#include <array>
#include <tuple>
#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2

// Usage: time ./smallpt 5000 && xv image.ppm

struct Vec {
  double x = 0.; // position, also color (r)
  double y = 0.; // position, also color (g)
  double z = 0.; // position, also color (b)

  Vec operator+(const Vec &b) const { return {x + b.x, y + b.y, z + b.z}; }
  Vec operator-(const Vec &b) const { return {x - b.x, y - b.y, z - b.z}; }
  Vec operator*(double b) const { return {x * b, y * b, z * b}; }
  Vec mult(const Vec &b) const { return {x * b.x, y * b.y, z * b.z}; }
  Vec &norm() { return *this = *this * (1 / sqrt(x * x + y * y + z * z)); }

  double dot(const Vec &b) const { return x * b.x + y * b.y + z * b.z; }

  // cross:
  Vec operator%(Vec &b) const {
    return {y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x};
  }
};

struct Ray {
  Vec o;
  Vec d;
};

enum class Refl { DIFF, SPEC, REFR }; // material types, used in radiance()

struct Sphere {
  double radius;
  Vec position;
  Vec emission;
  Vec color;
  Refl refl; // reflection type (DIFFuse, SPECular, REFRactive)

  double intersect(const Ray &r) const { // returns distance, 0 if nohit
    Vec op = position - r.o;             // Solve t^2*d.d + 2*t*(o-position).d +
                                         // (o-position).(o-position)-R^2 = 0
    double t;
    double eps = 1e-4;
    double b = op.dot(r.d);
    double det = b * b - op.dot(op) + radius * radius;

    if (det < 0) {
      return 0;
    } else {
      det = sqrt(det);
    }

    return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
  }
};

std::array<Sphere, 9> spheres = {{
    // Scene: radius, position, emission, color, material
    {1e5, {1e5 + 1, 40.8, 81.6}, {}, {.75, .25, .25}, Refl::DIFF},   // Left
    {1e5, {-1e5 + 99, 40.8, 81.6}, {}, {.25, .25, .75}, Refl::DIFF}, // Rght
    {1e5, {50, 40.8, 1e5}, {}, {.75, .75, .75}, Refl::DIFF},         // Back
    {1e5, {50, 40.8, -1e5 + 170}, {}, {}, Refl::DIFF},               // Frnt
    {1e5, {50, 1e5, 81.6}, {}, {.75, .75, .75}, Refl::DIFF},         // Botm
    {1e5, {50, -1e5 + 81.6, 81.6}, {}, {.75, .75, .75}, Refl::DIFF}, // Top
    {16.5, {27, 16.5, 47}, {}, Vec{1, 1, 1} * .999, Refl::SPEC},     // Mirr
    {16.5, {73, 16.5, 78}, {}, Vec{1, 1, 1} * .999, Refl::REFR},     // Glas
    {600, {50, 681.6 - .27, 81.6}, {12, 12, 12}, {}, Refl::DIFF}     // Lite
}};

inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

inline int toInt(double x) {
  return static_cast<int>(pow(clamp(x), 1 / 2.2) * 255 + .5);
}

inline std::pair<bool, Sphere *> intersect(const Ray &r, double &t) {
  double d;
  double inf = t = 1e20;
  Sphere *id;
  for (auto &s : spheres) {
    if ((d = s.intersect(r)) && d < t) {
      t = d;
      id = &s;
    }
  }
  return std::make_pair(t < inf, id);
}

Vec radiance(const Ray &r, int depth, unsigned short *Xi) {
  double t;   // distance to intersection
  Sphere *id; // id of intersected object
  bool intersected;
  std::tie(intersected, id) = intersect(r, t);
  if (!intersected) {
    return {}; // if miss, return black
  }

  const Sphere &obj = *id; // the hit object
  Vec x = r.o + r.d * t;
  Vec n = (x - obj.position).norm();
  Vec nl = n.dot(r.d) < 0 ? n : n * -1;
  Vec f = obj.color;
  double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl

  if (++depth > 5) {
    if (erand48(Xi) < p) {
      f = f * (1 / p);
    } else {
      return obj.emission; // R.R.
    }
  }

  if (obj.refl == Refl::DIFF) { // Ideal DIFFUSE reflection
    double r1 = 2 * M_PI * erand48(Xi);
    double r2 = erand48(Xi);
    double r2s = sqrt(r2);
    Vec w = nl;
    Vec u = ((fabs(w.x) > .1 ? Vec{0, 1} : Vec{1}) % w).norm();
    Vec v = w % u;
    Vec d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();
    return obj.emission + f.mult(radiance({x, d}, depth, Xi));
  } else if (obj.refl == Refl::SPEC) { // Ideal SPECULAR reflection
    return obj.emission +
           f.mult(radiance({x, r.d - n * 2 * n.dot(r.d)}, depth, Xi));
  }
  Ray reflRay = {x, r.d - n * 2 * n.dot(r.d)}; // Ideal dielectric REFRACTION
  bool into = n.dot(nl) > 0;                   // Ray from outside going in?
  double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl),
         cos2t;
  if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) <
      0) // Total internal reflection
    return obj.emission + f.mult(radiance(reflRay, depth, Xi));
  Vec tdir =
      (r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
  double a = nt - nc, b = nt + nc, R0 = a * a / (b * b),
         c = 1 - (into ? -ddn : tdir.dot(n));
  double Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re,
         RP = Re / P, TP = Tr / (1 - P);
  return obj.emission +
         f.mult(depth > 2
                    ? (erand48(Xi) < P ? // Russian roulette
                           radiance(reflRay, depth, Xi) * RP
                                       : radiance({x, tdir}, depth, Xi) * TP)
                    : radiance(reflRay, depth, Xi) * Re +
                          radiance({x, tdir}, depth, Xi) * Tr);
}
int main(int argc, char *argv[]) {
  int w = 1024, h = 768, samps = argc == 2 ? atoi(argv[1]) / 4 : 1; // # samples
  Ray cam{{50, 52, 295.6}, Vec{0, -0.042612, -1}.norm()}; // cam pos, dir
  Vec cx = Vec{w * .5135 / h}, cy = (cx % cam.d).norm() * .5135, r,
      *c = new Vec[w * h];
#pragma omp parallel for schedule(dynamic, 1) private(r) // OpenMP
  for (int y = 0; y < h; y++) {                          // Loop over image rows
    fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4,
            100. * y / (h - 1));
    for (unsigned short x = 0,
                        Xi[3] = {0, 0, static_cast<unsigned short>(y * y * y)};
         x < w; x++) // Loop cols
      for (int sy = 0, i = (h - y - 1) * w + x; sy < 2;
           sy++)                                    // 2x2 subpixel rows
        for (int sx = 0; sx < 2; sx++, r = Vec()) { // 2x2 subpixel cols
          for (int s = 0; s < samps; s++) {
            double r1 = 2 * erand48(Xi),
                   dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
            double r2 = 2 * erand48(Xi),
                   dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
            Vec d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
                    cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
            r = r + radiance({cam.o + d * 140, d.norm()}, 0, Xi) * (1. / samps);
          } // Camera rays are pushed ^^^^^ forward to start in interior
          c[i] = c[i] + Vec{clamp(r.x), clamp(r.y), clamp(r.z)} * .25;
        }
  }
  FILE *f = fopen("image.ppm", "w"); // Write image to PPM file.
  fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
  for (int i = 0; i < w * h; i++)
    fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
}
