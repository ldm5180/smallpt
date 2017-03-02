#include <algorithm>
#include <array>
#include <tuple>
#include <vector>
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

  double greatestPosition() const {
    if ((x > y) && (x > z)) {
      return x;
    }
    if (y > z) {
      return y;
    }
    return z;
  }

  // cross:
  Vec operator%(const Vec &b) const {
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

  double intersect(const Ray &r) const {
    // returns distance, 0 if nohit Solve
    // t^2*d.d + 2*t*(o-position).d + (o-position).(o-position)-R^2 = 0
    auto op = position - r.o;       

    constexpr double eps = 1e-4;
    auto b = op.dot(r.d);
    auto det = b * b - op.dot(op) + radius * radius;

    if (det < 0) {
      return 0;
    }
    det = sqrt(det);

    auto t = b - det;
    if (t > eps) {
      return t;
    }

    t = b + det;
    if (t > eps) {
      return t;
    }

    return 0;
  }
};

inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

inline int toInt(double x) {
  return static_cast<int>(pow(clamp(x), 1 / 2.2) * 255 + .5);
}

inline std::tuple<bool, Sphere *, double> intersect(const Ray &r) {
  static std::array<Sphere, 9> spheres = {{
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

  constexpr double inf = 1e20;
  auto t = inf;
  Sphere *id;
  std::for_each(spheres.rbegin(), spheres.rend(), [&](auto &s) {
    auto d = s.intersect(r);
    if (d && d < t) {
      t = d;
      id = &s;
    }
  });
  return std::make_tuple(t < inf, id, t);
}

Vec createNl(const Vec &n, const Ray &r) {
  return [&]() {
    if (n.dot(r.d) < 0) {
      return n;
    }
    return n * -1;
  }();
}

Vec radiance(const Ray &r, int depth, unsigned short *Xi);

Vec diffuseReflection(unsigned short *Xi, const Vec &nl, const Sphere &obj,
                      const Vec &f, const Vec &x, const int &depth) {
  double r1 = 2 * M_PI * erand48(Xi);
  double r2 = erand48(Xi);
  double r2s = sqrt(r2);
  Vec u = ((fabs(nl.x) > .1 ? Vec{0, 1} : Vec{1}) % nl).norm();
  Vec v = nl % u;
  Vec d = (u * cos(r1) * r2s + v * sin(r1) * r2s + nl * sqrt(1 - r2)).norm();
  return obj.emission + f.mult(radiance({x, d}, depth, Xi));
}

Vec radiance(const Ray &r, int depth, unsigned short *Xi) {
  double t;   // distance to intersection
  Sphere *id; // id of intersected object
  bool intersected;
  std::tie(intersected, id, t) = intersect(r);
  if (!intersected) {
    return {}; // if miss, return black
  }

  const Sphere &obj = *id; // the hit object
  Vec x = r.o + r.d * t;
  Vec n = (x - obj.position).norm();
  Vec nl = createNl(n, r);

  auto position = obj.color.greatestPosition();

  ++depth;

  Vec f = obj.color;
  if (depth > 5) {
    if (erand48(Xi) < position) {
      f = f * (1 / position);
    } else {
      return obj.emission; // R.R.
    }
  }

  if (obj.refl == Refl::DIFF) { // Ideal DIFFUSE reflection
    return diffuseReflection(Xi, nl, obj, f, x, depth);
  } else if (obj.refl == Refl::SPEC) { // Ideal SPECULAR reflection
    return obj.emission +
           f.mult(radiance({x, r.d - n * 2 * n.dot(r.d)}, depth, Xi));
  }

  Ray reflRay = {x, r.d - n * 2 * n.dot(r.d)}; // Ideal dielectric REFRACTION
  bool into = n.dot(nl) > 0;                   // Ray from outside going in?
  constexpr double nc = 1;
  constexpr double nt = 1.5;
  double nnt = into ? nc / nt : nt / nc;
  double ddn = r.d.dot(nl);

  double cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
  if (cos2t < 0) { // Total internal reflection
    return obj.emission + f.mult(radiance(reflRay, depth, Xi));
  }

  Vec tdir =
      (r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
  constexpr double a = nt - nc;
  constexpr double b = nt + nc;
  constexpr double R0 = a * a / (b * b);
  double c = 1 - (into ? -ddn : tdir.dot(n));
  double Re = R0 + (1 - R0) * c * c * c * c * c;
  double Tr = 1 - Re;
  double P = .25 + .5 * Re;
  double RP = Re / P;
  double TP = Tr / (1 - P);
  return obj.emission +
         f.mult(depth > 2
                    ? (erand48(Xi) < P ? // Russian roulette
                           radiance(reflRay, depth, Xi) * RP
                                       : radiance({x, tdir}, depth, Xi) * TP)
                    : radiance(reflRay, depth, Xi) * Re +
                          radiance({x, tdir}, depth, Xi) * Tr);
}
int main(int argc, char *argv[]) {
  constexpr int width = 1024;
  constexpr int height = 768;
  int samples = argc == 2 ? atoi(argv[1]) / 4 : 1;        // # samples
  Ray camera{{50, 52, 295.6}, Vec{0, -0.042612, -1}.norm()}; // camera pos, dir
  Vec cx = Vec{width * .5135 / height};
  Vec cy = (cx % camera.d).norm() * .5135;
  Vec r;
  std::vector<Vec> c(width * height);

#pragma omp parallel for schedule(dynamic, 1) private(r) // OpenMP

  for (int y = 0; y < height; y++) { // Loop over image rows
    fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samples * 4,
            100. * y / (height - 1));
    for (unsigned short x = 0,
                        Xi[3] = {0, 0, static_cast<unsigned short>(y * y * y)};
         x < width; x++) // Loop cols
      for (int sy = 0, i = (height - y - 1) * width + x; sy < 2;
           sy++)                                    // 2x2 subpixel rows
        for (int sx = 0; sx < 2; sx++, r = Vec()) { // 2x2 subpixel cols
          for (int s = 0; s < samples; s++) {
            double r1 = 2 * erand48(Xi),
                   dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
            double r2 = 2 * erand48(Xi),
                   dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
            Vec d = cx * (((sx + .5 + dx) / 2 + x) / width - .5) +
                    cy * (((sy + .5 + dy) / 2 + y) / height - .5) + camera.d;
            auto mult140 = camera.o + d * 140;
            r = r + radiance({mult140, d.norm()}, 0, Xi) * (1. / samples);
          } // Camera rays are pushed ^^^^^ forward to start in interior
          c[i] = c[i] + Vec{clamp(r.x), clamp(r.y), clamp(r.z)} * .25;
        }
  }
  FILE *f = fopen("image.ppm", "w"); // Write image to PPM file.
  fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
  for (const auto &vec : c) {
    fprintf(f, "%d %d %d ", toInt(vec.x), toInt(vec.y), toInt(vec.z));
  }
}
