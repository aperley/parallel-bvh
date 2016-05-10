#include "bvh.h"

#include "CMU462/CMU462.h"
#include "static_scene/triangle.h"
#include "CycleTimer.h"
#include <omp.h>

#include <mutex>
#include <thread>

#include <iostream>
#include <stack>
#include <cassert>

using namespace std;

namespace CMU462 { namespace StaticScene {

#define NUM_BUCKETS 12

BVHAccel::BVHAccel(const std::vector<Primitive *> &_primitives,
                   size_t numBVHThreads, size_t max_leaf_size) {

  this->primitives = _primitives;
  printf("%d primitives\n", primitives.size());
  omp_set_nested(1);  // need nested parallel calls
  omp_set_dynamic(0); // want team size set by calls to set_num_threads
  omp_set_num_threads(numBVHThreads);

  #pragma omp parallel
  {
    #pragma omp single
      printf("OPENMP: num_threads=%d, nested=%d, dynamic=%d\n",
           omp_get_num_threads(), omp_get_nested(), omp_get_dynamic());
  }

  // Construct a BVH from the given vector of primitives and maximum leaf
  // size configuration. The starter code build a BVH aggregate with a
  // single leaf node (which is also the root) that encloses all the
  // primitives.

  double start = CycleTimer::currentSeconds();
  BBox bb;
  #pragma omp parallel
  {
    BBox pbb;

    #pragma omp for
    for (size_t i = 0; i < primitives.size(); ++i) {
      pbb.expand(primitives[i]->get_bbox());
    }

    #pragma omp critical
    {
      bb.expand(pbb);
    }
  }

  root = new BVHNode(bb, 0, primitives.size());
  partition(*root, max_leaf_size, numBVHThreads);

  double end = CycleTimer::currentSeconds();

  printf("BVH Build: %f seconds\n", end - start);

}


void BVHAccel::partition(BVHNode& node, size_t max_leaf_size, int num_threads) {
  omp_set_num_threads(num_threads);

  if (node.range <= max_leaf_size)
    return;

  double node_sa_recip = 1.0 / node.bb.surface_area();

  double best_sah = INF_D;
  BBox best_left_bbox, best_right_bbox;
  size_t best_left_count, best_right_count;
  int best_axis;
  int best_pplane;
  double best_cmax, best_cmin;

  for (int axis = 0; axis < 3; ++axis) {
    BBox bboxes[NUM_BUCKETS] = {BBox()};
    int prim_counts[NUM_BUCKETS] = {0};

    // Compute centroid min and centroid_max
    double centroid_min = INF_D;
    double centroid_max = -INF_D;
    #pragma omp parallel for reduction(min : centroid_min), reduction(max : centroid_max)
    for (int i = node.start; i < node.start + node.range; ++i) {
      Primitive* p = primitives[i];
      double centroid = p->get_bbox().centroid()[axis];
      if (centroid < centroid_min)
        centroid_min = centroid;
      if (centroid > centroid_max)
        centroid_max = centroid;
    }

    // Do not try to split along an axis that has 0 extent
    if (centroid_max - centroid_min < EPS_D)
      continue;

    // Put primitives in buckets, get bounding boxes and prim counts
    #pragma omp parallel
    {
      BBox pbboxes[NUM_BUCKETS] = {BBox()};
      int pprim_counts[NUM_BUCKETS] = {0};

      #pragma omp for
      for (int i = node.start; i < node.start + node.range; ++i) {
        Primitive* p = primitives[i];
        BBox p_bbox = p->get_bbox();
        double centroid = p_bbox.centroid()[axis];
        int bucket = compute_bucket(centroid_min, centroid_max, centroid);
        pbboxes[bucket].expand(p_bbox);
        ++pprim_counts[bucket];
      }

      #pragma omp critical
      for (int i = 0; i < NUM_BUCKETS; ++i) {
        bboxes[i].expand(pbboxes[i]);
        prim_counts[i] += pprim_counts[i];
      }
    }


    // Find the best partitioning plane for this axis
    for (int pplane = 1; pplane < NUM_BUCKETS; ++pplane) {
      BBox left_bbox, right_bbox;
      int left_count = 0, right_count = 0;

      for (int j = 0; j < pplane; ++j) {
        left_bbox.expand(bboxes[j]);
        left_count += prim_counts[j];
      }

      for (int j = pplane; j < NUM_BUCKETS; ++j) {
        right_bbox.expand(bboxes[j]);
        right_count += prim_counts[j];
      }

      if (left_count == 0 || right_count == 0)
        continue;

      // we are only interested in relative sahs, so we can remove
      // constant parts of the sah, including the surface area of the
      // node's bounding box
      double sah = left_bbox.surface_area() * node_sa_recip*left_count +
                right_bbox.surface_area() * node_sa_recip * right_count;

      if (sah < best_sah) {
        best_sah = sah;
        best_left_bbox = left_bbox;
        best_right_bbox = right_bbox;
        best_left_count = left_count;
        best_right_count = right_count;
        best_axis = axis;
        best_pplane = pplane;
        best_cmax = centroid_max;
        best_cmin = centroid_min;
      }
    }
  }

  if (best_sah == INF_D) {
    // The centroids were all the same
    BBox bb_l, bb_r;
    for (int j = node.start; j < node.start + node.range / 2; ++j) {
      bb_l.expand(primitives[j]->get_bbox());
    }
    for (int j = node.start + node.range / 2; j < node.start + node.range; ++j) {
      bb_r.expand(primitives[j]->get_bbox());
    }
    node.l = new BVHNode(bb_l, node.start, node.range / 2);
    node.r = new BVHNode(bb_r, node.start + node.range / 2, node.range / 2);
  }
  else {
    // Now do the partitioning
    auto prim_in_left =
      [&best_axis, &best_pplane, &best_cmin, &best_cmax](Primitive*& prim) {
      double centroid = prim->get_bbox().centroid()[best_axis];
      return compute_bucket(best_cmin, best_cmax, centroid) < best_pplane;
    };

    std::partition(primitives.begin() + node.start,
                   primitives.begin() + node.start + node.range,
                   prim_in_left);

    node.l = new BVHNode(best_left_bbox, node.start, best_left_count);
    node.r = new BVHNode(best_right_bbox, node.start + best_left_count, best_right_count);
  }

    if (num_threads > 1) {
      #pragma omp parallel num_threads(2)
      {
        if (omp_get_thread_num() == 0) {
          // Round up if we have an odd number
          if (num_threads % 2 == 1) {
            num_threads++;
          }
          partition(*node.l, max_leaf_size, num_threads/2);
        }
        else {
          partition(*node.r, max_leaf_size, num_threads/2);
        }
      }
    }
    else {
      partition(*node.l, max_leaf_size, 1);
      partition(*node.r, max_leaf_size, 1);
    }

  return;
}

inline int BVHAccel::compute_bucket(double centroid_min, double centroid_max, double centroid)
{
  int result = (int) floor((centroid - centroid_min) / (centroid_max - centroid_min + EPS_D) * NUM_BUCKETS);
  if (!(0 <= result && result < NUM_BUCKETS)) {
    printf("aaaaah!\n");
    assert(false);
  }
  return result;
}

BVHAccel::~BVHAccel() {

  // Implement a proper destructor for your BVH accelerator aggregate
  // The BVH node destructor (added by me) will recursively delete its
  // child nodes
  delete root;

}

BBox BVHAccel::get_bbox() const {
  return root->bb;
}

bool BVHAccel::intersect(const Ray &ray) const {

  // Implement ray - bvh aggregate intersection test. A ray intersects
  // with a BVH aggregate if and only if it intersects a primitive in
  // the BVH that is not an aggregate.
  Intersection i;
  return intersect(ray, &i);
}

bool BVHAccel::intersect(const Ray &ray, Intersection *i) const {

  // Implement ray - bvh aggregate intersection test. A ray intersects
  // with a BVH aggregate if and only if it intersects a primitive in
  // the BVH that is not an aggregate. When an intersection does happen.
  // You should store the non-aggregate primitive in the intersection data
  // and not the BVH aggregate itself.

  Intersection test_i;
  find_closest_hit(ray, *root, test_i);
  if (test_i.t < INF_D) {
    *i = test_i;
    return true;
  }

  return false;
}

void BVHAccel::find_closest_hit(const Ray &ray, const BVHNode &node, Intersection &i) const {

  double box_t0 = -INF_D, box_t1 = INF_D;
  if (!node.bb.intersect(ray, box_t0, box_t1) || box_t0 > i.t)
    return;

  if (node.isLeaf()) {
    Intersection current_i;
    for (size_t p_i = node.start; p_i < node.start + node.range; ++p_i) {
      if (primitives[p_i]->intersect(ray, &current_i)) {
        if (0 <= current_i.t && current_i.t < i.t) {
          i = current_i;
        }
      }
    }
  } else {
    double t_l0 = -INF_D, t_l1 = INF_D;
    bool hit_l = node.l->bb.intersect(ray, t_l0, t_l1);
    double t_r0 = -INF_D, t_r1 = INF_D;
    bool hit_r = node.r->bb.intersect(ray, t_r0, t_r1);

    BVHNode* first = (t_l0 <= t_r0) ? node.l : node.r;
    BVHNode* second = (t_l0 <= t_r0) ? node.r : node.l;

    find_closest_hit(ray, *first, i);
    if (t_r0 < i.t)
      find_closest_hit(ray, *second, i);
  }
}

}  // namespace StaticScene
}  // namespace CMU462
