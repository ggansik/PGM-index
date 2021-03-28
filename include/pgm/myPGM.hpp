//
// Created by Hyunsik Yoon on 2021-03-25.
//

#pragma once

#include <limits>
#include <vector>
#include <utility>
#include <algorithm>
#include "piecewise_linear_model.hpp"

namespace myPGM {

#define PGM_SUB_EPS(x, epsilon) ((x) <= (epsilon) ? 0 : ((x) - (epsilon)))
#define PGM_ADD_EPS(x, epsilon, size) ((x) + (epsilon) + 2 >= (size) ? (size) : (x) + (epsilon) + 2)

struct ApproxPos {
    size_t pos; ///< The approximate position of the key.
    size_t lo;  ///< The lower bound of the range.
    size_t hi;  ///< The upper bound of the range.
};

template<typename K, size_t Epsilon = 64, size_t EpsilonRecursive = 4, typename Floating = float>
class PGMIndex {
protected:
    template<typename, size_t, uint8_t, typename>
    friend class BucketingPGMIndex;

    template<typename, size_t, typename>
    friend class EliasFanoPGMIndex;

    static_assert(Epsilon > 0);
    struct Segment;

    size_t n;                           ///< The number of elements this index was built on.
    K first_key;                        ///< The smallest element.
    std::vector<Segment> segments;      ///< The segments composing the index.
    std::vector<size_t> levels_sizes;   ///< The number of segment in each level, in reverse order.
    std::vector<size_t> levels_offsets; ///< The starting position of each level in segments[], in reverse order. 레벨 구분 없이 세그먼트가 segments 벡터에 들어가 있음.

    template<typename RandomIt>
    static void build(RandomIt first, RandomIt last,
                      size_t epsilon, size_t epsilon_recursive,
                      std::vector<Segment> &segments,
                      std::vector<size_t> &levels_sizes,
                      std::vector<size_t> &levels_offsets) {
        auto n = std::distance(first, last);
        if (n == 0) //data 수가 0이면
            return;

        levels_offsets.push_back(0);
        segments.reserve(n / (epsilon * epsilon)); //최대 세그먼트 수?

        auto ignore_last = *std::prev(last) == std::numeric_limits<K>::max(); // max is reserved for padding 배열에서 제일 큰 값이 K자료형의 최대값이라면  마지막거는 무시
        auto last_n = n - ignore_last;
        last -= ignore_last;

        auto build_level = [&](auto epsilon, auto in_fun, auto out_fun) {
            auto n_segments = internal::make_segmentation_par(last_n, epsilon, in_fun, out_fun); //parallel하게 만듦 optimal 세그먼트 수를 계산해주는듯
            if (segments.back().slope == 0) {//이부분은 머지
                // Here we need to ensure that keys > *(last-1) are approximated to a position == prev_level_size
                segments.emplace_back(*std::prev(last) + 1, 0, last_n);
                ++n_segments;
            }
            segments.emplace_back(last_n);
            return n_segments;
        };

        // Build first level
        auto in_fun = [&](auto i) {
            auto x = first[i];
            // Here there is an adjustment for inputs with duplicate keys: at the end of a run of duplicate keys equal
            // to x=first[i] such that x+1!=first[i+1], we map the values x+1,...,first[i+1]-1 to their correct rank i
            auto flag = i > 0 && i + 1u < n && x == first[i - 1] && x != first[i + 1] && x + 1 != first[i + 1];
            return std::pair<K, size_t>(x + flag, i);
        };
        auto out_fun = [&](auto cs) { segments.emplace_back(cs); };
        last_n = build_level(epsilon, in_fun, out_fun);
        levels_offsets.push_back(levels_offsets.back() + last_n + 1);
        levels_sizes.push_back(last_n);

        // Build upper levels
        while (epsilon_recursive && last_n > 1) {
            auto offset = levels_offsets[levels_offsets.size() - 2];
            auto in_fun_rec = [&](auto i) { return std::pair<K, size_t>(segments[offset + i].key, i); };
            last_n = build_level(epsilon_recursive, in_fun_rec, out_fun);
            levels_offsets.push_back(levels_offsets.back() + last_n + 1);
            levels_sizes.push_back(last_n);
        }

        levels_offsets.pop_back();
    }

    /**
     * Returns the segment responsible for a given key, that is, the rightmost segment having key <= the sought key.
     * @param key the value of the element to search for
     * @return an iterator to the segment responsible for the given key
     */
    auto segment_for_key(const K &key) const {
        if constexpr (EpsilonRecursive == 0) {
            auto it = std::upper_bound(segments.begin(), segments.begin() + levels_sizes[0], key);
            return it == segments.begin() ? it : std::prev(it);
        }

        auto it = segments.begin() + levels_offsets.back();

        for (auto l = int(height()) - 2; l >= 0; --l) {
            auto level_begin = segments.begin() + levels_offsets[l];
            auto pos = std::min<size_t>((*it)(key), std::next(it)->intercept);
            auto lo = level_begin + PGM_SUB_EPS(pos, EpsilonRecursive + 1);

            static constexpr size_t linear_search_threshold = 8 * 64 / sizeof(Segment);
            if constexpr (EpsilonRecursive <= linear_search_threshold) {
                for (; std::next(lo)->key <= key; ++lo)
                    continue;
                it = lo;
            } else {
                auto level_size = levels_sizes[l];
                auto hi = level_begin + PGM_ADD_EPS(pos, EpsilonRecursive, level_size);
                it = std::upper_bound(lo, hi, key);
                it = it == level_begin ? it : std::prev(it);
            }
        }
        return it;
    }

public:

    static constexpr size_t epsilon_value = Epsilon;

    /**
     * Constructs an empty index.
     */
    PGMIndex() = default;

    /**
     * Constructs the index on the given sorted vector.
     * @param data the vector of keys to be indexed, must be sorted
     */
    //생성자, explicit 키워드는 명시적 생성자만 호출가능하도록 함
    explicit PGMIndex(const std::vector<K> &data) : PGMIndex(data.begin(), data.end()) {}

    /**
     * Constructs the index on the sorted keys in the range [first, last).
     * @param first, last the range containing the sorted keys to be indexed
     */
    template<typename RandomIt>
    //생성자
    PGMIndex(RandomIt first, RandomIt last)
            : n(std::distance(first, last)),
              first_key(n ? *first : 0),
              segments(),
              levels_sizes(),
              levels_offsets() {
        build(first, last, Epsilon, EpsilonRecursive, segments, levels_sizes, levels_offsets);
    }
    /**
     * 현식수정
     */
    void insert(const K &key){
        //이 모든게 가능하기 위한 전제? array가 segment단위로 잘려서 linked list 형태로 저장되어야 함. 이로 인한 오버헤드 생각? 아 근데 버퍼에 저장되면 노상관인데?
        //segment마다 first Key, slope, intercept를 저장하고 있음
        //방법 1. segment마다 레코드가 삽입된 위치를 기억(bit연산 사용가능한지 확인하기 비트로 표현 가능.. 근데 같은 위치에 계속 들어오면?...)
        //방법 2. segment마다 삽입된 레코드 수만 파악하고, 확률적으로 계산->considerations for handling updates~

        /**
         * 1. key가 삽입될 segment를 찾는다
         * 2. segment내에서 삽입될 위치를 찾는다
         * search 를 어떻게 할 것인가?
         * 2-1. model-based로 삽입될 위치를 찾고 거기부터 Local search 한다
         * 2-2. 처음부터 binary search 한다
         * 2-3. exponential search 한다
         * 3. 해당 위치를 표시하고 insert한다
         * insert를 어떻게 할 것인가?
         * 3-1-1. buffer에 insert하고 표시를 가지고 buffer 내부 위치를 계산한다(이거 가능한가? 근데 버퍼에 넣을거면 내가 생각했던 prediction correction이 필요가 없어지는것 아닌가?)
         *      -> 원래 array에서 insert되어야 하는 위치보다 앞에있는 비트 1의 수를 세면 버퍼 내부의 위치를 알 수 있을 것 같은데?
         *      -> 근데 그럼 search할 때 원래 array를 보고서 없는 경우에 추가적으로 버퍼도 참조해야 함. 근데 delta 보단 빠른게 delta는 버퍼 전체에 대한 search를 해야 했음)
         * 3-1-2. 원래 array에 껴넣고 데이터 전부 다 shift... shift비용은 얼마나 되는지?
         * 위치 표시는 어떻게 할 것인가?
         * 3-2-1. 세그먼트마다 비트를 유지한다.
         * 3-2-2.
         * 4. 에러 바운드와 버퍼 내부 내용의 병합? 버퍼에 쌓일수록 성능이 저하될 것이고(, 같은 곳에 계속 삽입되면 그걸 표현하기도 힘들고,
         * 5.
         */
    }
    /**
     * Returns the approximate position and the range where @p key can be found.
     * @param key the value of the element to search for
     * @return a struct with the approximate position and bounds of the range
     */
    ApproxPos search(const K &key) const {
        auto k = std::max(first_key, key); //이건 왜함
        auto it = segment_for_key(k); //key를 포함하는 세그먼트를 찾아서
        auto pos = std::min<size_t>((*it)(k), std::next(it)->intercept); //f(k)나 해당 segment의 intercept = 아마도 다음 segment의 시작 key?
        auto lo = PGM_SUB_EPS(pos, Epsilon);
        auto hi = PGM_ADD_EPS(pos, Epsilon, n);
        return {pos, lo, hi};
    }

    /**
     * Returns the number of segments in the last level of the index.
     * @return the number of segments
     */
    size_t segments_count() const {
        return segments.empty() ? 0 : levels_sizes.front();
    }

    /**
     * Returns the number of levels of the index.
     * @return the number of levels of the index
     */
    size_t height() const {
        return levels_sizes.size();
    }

    /**
     * Returns the size of the index in bytes.
     * @return the size of the index in bytes
     */
    size_t size_in_bytes() const {
        return segments.size() * sizeof(Segment);
    }
};

#pragma pack(push, 1)

template<typename K, size_t Epsilon, size_t EpsilonRecursive, typename Floating>
struct PGMIndex<K, Epsilon, EpsilonRecursive, Floating>::Segment {
    K key;             ///< The first key that the segment indexes.
    Floating slope;    ///< The slope of the segment.
    int32_t intercept; ///< The intercept of the segment.

    Segment() = default;

    Segment(K key, Floating slope, Floating intercept) : key(key), slope(slope), intercept(intercept) {};

    explicit Segment(size_t n) : key(std::numeric_limits<K>::max()), slope(), intercept(n) {};

    explicit Segment(const typename internal::OptimalPiecewiseLinearModel<K, size_t>::CanonicalSegment &cs)
            : key(cs.get_first_x()) {
        auto[cs_slope, cs_intercept] = cs.get_floating_point_segment(key);
        if (cs_intercept > std::numeric_limits<decltype(intercept)>::max())
            throw std::overflow_error("Change the type of Segment::intercept to int64");
        slope = cs_slope;
        intercept = std::round(cs_intercept);
    }

    friend inline bool operator<(const Segment &s, const K &k) { return s.key < k; }
    friend inline bool operator<(const K &k, const Segment &s) { return k < s.key; }
    friend inline bool operator<(const Segment &s, const Segment &t) { return s.key < t.key; }

    operator K() { return key; };

    /**
     * Returns the approximate position of the specified key.
     * @param k the key whose position must be approximated
     * @return the approximate position of the specified key
     */
    //segment(key)이렇게 사용할 수 있도록 만들어 주는 operator를 정의하는듯
    inline size_t operator()(const K &k) const {
        auto pos = int64_t(slope * (k - key)) + intercept;
        return pos > 0 ? size_t(pos) : 0ull;
    }
};

#pragma pack(pop)

}