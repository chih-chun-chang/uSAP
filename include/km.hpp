// https://github.com/hpec-graphchallenge/BlockFinder/blob/master/blockfinder/km.hpp
//
//

#pragma once

#include <iostream>
#include <vector>
#include <limits>
#include <cassert>

namespace bf {

// Class: KuhnNMunkresSolver
//
// This class implements the KM assignment algorithm (Hungarian algorithm).
//
template <typename W>
class KuhnMunkresSolver {

  public:

    KuhnMunkresSolver() = default;

    void solve(const std::vector<std::vector<W>>& cost);

    const std::vector<int>& match1() const;
    const std::vector<int>& match2() const;

  private:

    std::vector<int> _s, _t, _l1, _l2;
    std::vector<int> _match1;
    std::vector<int> _match2;

};

template <typename W>
const std::vector<int>& KuhnMunkresSolver<W>::match1() const {
  return _match1;
}

template <typename W>
const std::vector<int>& KuhnMunkresSolver<W>::match2() const {
  return _match2;
}

template <typename W>
void KuhnMunkresSolver<W>::solve(const std::vector<std::vector<W>>& cost){

  if(cost.size() == 0) {
    return;
  }

  int _m = cost.size();
  int _n = cost[0].size();

  _s.resize(_m);
  _t.resize(_n);
  _l1.resize(_m);
  _l2.resize(_n);
  _match1.resize(_m);
  _match2.resize(_n);

  _match1.assign(_m, std::numeric_limits<int>::min());
  _match2.assign(_n, std::numeric_limits<int>::min());
  _l2.assign(_n, 0);

  int p, q, i, j, k;

  for(i=0;i<_m;i++) {
    for(_l1[i]=std::numeric_limits<int>::min(), j=0; j<_n; j++) {
      _l1[i] = cost[i][j] > _l1[i] ? cost[i][j] : _l1[i];
    }
  }

  // augmented path from i
  for(i=0;i<_m;i++) {

    _t.assign(_n, std::numeric_limits<int>::min());

    for(_s[p=q=0]=i; p<=q && _match1[i]<0; p++) {
      for(k=_s[p],j=0; j<_n && _match1[i]<0; j++) {
        if(_l1[k]+_l2[j] == cost[k][j] && _t[j]<0) {
          _s[++q] = _match2[j];
          _t[j] = k;
          if(_s[q]<0) {
            for(p=j;p>=0;j=p) {
              _match2[j]=k=_t[j];
              p = _match1[k];
              _match1[k] = j;
            }
          }
        }
      }
    }

    // path not found, adjust l1 and l2 weight
    if(_match1[i]<0) {
      // nodes with t[j]>0 and not in s are min vertex cover
      for(i--,p=std::numeric_limits<int>::max(),k=0;k<=q;k++) {
        for(j=0;j<_n;j++) {
          // for cost that is in s but not in t, find min cost
          if(_t[j]<0 && _l1[_s[k]] + _l2[j] - cost[_s[k]][j]<p) {
            p=_l1[_s[k]]+_l2[j]-cost[_s[k]][j];
          }
        }
      }
      // for l2 in t, +p
      for(j=0; j<_n; _l2[j] += _t[j]<0 ? 0 : p, j++);
      // for l1 in s, -p
      for(k=0; k<=q; _l1[_s[k++]]-=p);
    }
  }
}


}  // end of namespace bf -----------------------------------------------------
