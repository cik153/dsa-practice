import java.io.*;
import java.util.*;

/**
 * 코테 베이스 템플릿 (요즘 5문제 코테에 자주 나오는 것만)
 * - FastScanner
 * - Hash/Sorting scan helpers
 * - BFS/DFS (grid/graph)
 * - Dijkstra
 * - DP skeleton
 * - Backtracking skeleton (pruning + duplicate handling)
 * - Two pointers / Sliding window
 * - Prefix sum + HashMap count
 * - Binary search (parametric)
 * - Heap(topK / scheduling)
 * - Fenwick / Segment Tree
 * - DSU
 *
 * 사용법:
 * - 이 파일에서 필요한 부분만 복사해서 제출용 Main.java에 붙여넣기
 */
public class Snippets {

    // =========================================================
    // 0) Fast I/O
    // =========================================================
    static class FastScanner {
        private final InputStream in;
        private final byte[] buffer = new byte[1 << 16];
        private int ptr = 0, len = 0;

        FastScanner(InputStream in) { this.in = in; }

        private int readByte() throws IOException {
            if (ptr >= len) {
                len = in.read(buffer);
                ptr = 0;
                if (len <= 0) return -1;
            }
            return buffer[ptr++];
        }

        long nextLong() throws IOException {
            int c;
            do { c = readByte(); } while (c <= ' ' && c != -1);
            if (c == -1) return Long.MIN_VALUE;

            long sign = 1;
            if (c == '-') { sign = -1; c = readByte(); }

            long val = 0;
            while (c > ' ') {
                val = val * 10 + (c - '0');
                c = readByte();
            }
            return val * sign;
        }

        int nextInt() throws IOException { return (int) nextLong(); }

        String next() throws IOException {
            int c;
            do { c = readByte(); } while (c <= ' ' && c != -1);
            if (c == -1) return null;

            StringBuilder sb = new StringBuilder();
            while (c > ' ') {
                sb.append((char) c);
                c = readByte();
            }
            return sb.toString();
        }
    }

    static final long INF = (long) 4e18;

    // =========================================================
    // 1) HashMap/HashSet: 대표 패턴 예시 (카운팅)
    // =========================================================
    static Map<Integer, Integer> countFreq(int[] arr) {
        HashMap<Integer, Integer> mp = new HashMap<>();
        for (int x : arr) mp.put(x, mp.getOrDefault(x, 0) + 1);
        return mp;
    }

    // =========================================================
    // 2) Sorting + scan: 겹침 최대(라인스위프 맛) 기본형
    //    events: (time, +1 start) / (time, -1 end) 형태로 정렬 후 누적
    // =========================================================
    static int maxOverlap(List<int[]> events) {
        // events: [time, delta], end 처리 규칙은 문제 정의에 맞춰 조정
        events.sort((a, b) -> {
            if (a[0] != b[0]) return Integer.compare(a[0], b[0]);
            return Integer.compare(a[1], b[1]); // 보통 end(-1)를 start(+1)보다 먼저/나중 조정
        });
        int cur = 0, ans = 0;
        for (int[] e : events) {
            cur += e[1];
            ans = Math.max(ans, cur);
        }
        return ans;
    }

    // =========================================================
    // 3) BFS/DFS (Graph)
    // =========================================================
    static int[] bfsUnweighted(int n, List<Integer>[] g, int start) {
        int[] dist = new int[n];
        Arrays.fill(dist, Integer.MAX_VALUE);
        ArrayDeque<Integer> q = new ArrayDeque<>();
        dist[start] = 0;
        q.add(start);

        while (!q.isEmpty()) {
            int u = q.poll();
            int du = dist[u];
            for (int v : g[u]) {
                if (dist[v] == Integer.MAX_VALUE) {
                    dist[v] = du + 1;
                    q.add(v);
                }
            }
        }
        return dist;
    }

    static int countConnectedComponents(int n, List<Integer>[] g) {
        boolean[] vis = new boolean[n];
        int comps = 0;
        ArrayDeque<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            if (vis[i]) continue;
            comps++;
            vis[i] = true;
            q.add(i);
            while (!q.isEmpty()) {
                int u = q.poll();
                for (int v : g[u]) {
                    if (!vis[v]) {
                        vis[v] = true;
                        q.add(v);
                    }
                }
            }
        }
        return comps;
    }

    // =========================================================
    // 4) BFS (Grid 4-dir): 최단 횟수 / 섬 개수
    // =========================================================
    static int[][] bfsGrid4(char[][] grid, int sr, int sc, boolean[][] passable) {
        int R = grid.length, C = grid[0].length;
        int[][] dist = new int[R][C];
        for (int i = 0; i < R; i++) Arrays.fill(dist[i], Integer.MAX_VALUE);

        int[] dr = {-1, 1, 0, 0};
        int[] dc = {0, 0, -1, 1};

        ArrayDeque<int[]> q = new ArrayDeque<>();
        dist[sr][sc] = 0;
        q.add(new int[]{sr, sc});

        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int r = cur[0], c = cur[1];
            int d = dist[r][c];
            for (int k = 0; k < 4; k++) {
                int nr = r + dr[k], nc = c + dc[k];
                if (0 <= nr && nr < R && 0 <= nc && nc < C
                        && passable[nr][nc] && dist[nr][nc] == Integer.MAX_VALUE) {
                    dist[nr][nc] = d + 1;
                    q.add(new int[]{nr, nc});
                }
            }
        }
        return dist;
    }

    static int countIslands(char[][] grid, char landChar) {
        int R = grid.length, C = grid[0].length;
        boolean[][] vis = new boolean[R][C];
        int[] dr = {-1, 1, 0, 0};
        int[] dc = {0, 0, -1, 1};
        ArrayDeque<int[]> q = new ArrayDeque<>();
        int islands = 0;

        for (int i = 0; i < R; i++) for (int j = 0; j < C; j++) {
            if (vis[i][j] || grid[i][j] != landChar) continue;
            islands++;
            vis[i][j] = true;
            q.add(new int[]{i, j});
            while (!q.isEmpty()) {
                int[] cur = q.poll();
                int r = cur[0], c = cur[1];
                for (int k = 0; k < 4; k++) {
                    int nr = r + dr[k], nc = c + dc[k];
                    if (0 <= nr && nr < R && 0 <= nc && nc < C
                            && !vis[nr][nc] && grid[nr][nc] == landChar) {
                        vis[nr][nc] = true;
                        q.add(new int[]{nr, nc});
                    }
                }
            }
        }
        return islands;
    }

    // =========================================================
    // 5) Dijkstra (weighted shortest path, w>=0)
    // =========================================================
    static class Edge {
        int to, w;
        Edge(int to, int w) { this.to = to; this.w = w; }
    }

    static long[] dijkstra(int n, List<Edge>[] g, int start) {
        long[] dist = new long[n];
        Arrays.fill(dist, INF);
        dist[start] = 0;

        PriorityQueue<long[]> pq = new PriorityQueue<>(Comparator.comparingLong(a -> a[0]));
        pq.add(new long[]{0, start});

        while (!pq.isEmpty()) {
            long[] cur = pq.poll();
            long d = cur[0];
            int u = (int) cur[1];
            if (d != dist[u]) continue;

            for (Edge e : g[u]) {
                long nd = d + e.w;
                if (nd < dist[e.to]) {
                    dist[e.to] = nd;
                    pq.add(new long[]{nd, e.to});
                }
            }
        }
        return dist;
    }

    // =========================================================
    // 6) DP skeleton (상태 정의용 기본 틀)
    // =========================================================
    // 예: dp[i] = i까지 봤을 때 최댓값/방법 수 등
    static long[] dp1DExample(int n) {
        long[] dp = new long[n + 1];
        Arrays.fill(dp, -1);
        dp[0] = 0;
        for (int i = 1; i <= n; i++) {
            // dp[i] = ...
        }
        return dp;
    }

    // =========================================================
    // 7) Backtracking (요즘 자주) - 핵심 3종 베이스
    //    (A) 조합/부분집합 (중복 없음)
    //    (B) 순열 (방문 배열)
    //    (C) 중복 원소 처리(같은 값 스킵)
    // =========================================================

    // (A) Choose k from nums (combinations), pruning included
    static List<int[]> chooseK(int[] nums, int k) {
        List<int[]> res = new ArrayList<>();
        int[] path = new int[k];
        dfsChoose(0, 0, nums, k, path, res);
        return res;
    }
    static void dfsChoose(int idx, int depth, int[] nums, int k, int[] path, List<int[]> res) {
        if (depth == k) {
            res.add(path.clone());
            return;
        }
        if (idx == nums.length) return;
        // pruning: remaining < needed
        if (nums.length - idx < k - depth) return;

        // take
        path[depth] = nums[idx];
        dfsChoose(idx + 1, depth + 1, nums, k, path, res);

        // skip
        dfsChoose(idx + 1, depth, nums, k, path, res);
    }

    // (B) Permutations of length k
    static List<int[]> permuteK(int[] nums, int k) {
        List<int[]> res = new ArrayList<>();
        int[] path = new int[k];
        boolean[] used = new boolean[nums.length];
        dfsPermute(0, nums, k, used, path, res);
        return res;
    }
    static void dfsPermute(int depth, int[] nums, int k, boolean[] used, int[] path, List<int[]> res) {
        if (depth == k) {
            res.add(path.clone());
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) continue;
            used[i] = true;
            path[depth] = nums[i];
            dfsPermute(depth + 1, nums, k, used, path, res);
            used[i] = false;
        }
    }

    // (C) Duplicate-handling permutations (nums must be sorted)
    // 대표: 같은 숫자가 여러 개 있을 때 중복 순열 제거
    static List<int[]> permuteAllUnique(int[] nums) {
        Arrays.sort(nums);
        List<int[]> res = new ArrayList<>();
        int[] path = new int[nums.length];
        boolean[] used = new boolean[nums.length];
        dfsPermuteUnique(0, nums, used, path, res);
        return res;
    }
    static void dfsPermuteUnique(int depth, int[] nums, boolean[] used, int[] path, List<int[]> res) {
        if (depth == nums.length) {
            res.add(path.clone());
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) continue;
            // 핵심: 같은 값은 "이전 동일 값이 이번 depth에서 아직 사용되지 않았으면" 스킵
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) continue;

            used[i] = true;
            path[depth] = nums[i];
            dfsPermuteUnique(depth + 1, nums, used, path, res);
            used[i] = false;
        }
    }

    // 백트래킹 가지치기 팁(베이스):
    // 1) 현재 cost >= bestCost 이면 return
    // 2) 남은 것으로 목표 달성 불가하면 return
    // 3) 동일 상태 중복이면 memo/set으로 컷

    // =========================================================
    // 8) Two Pointers / Sliding Window (비음수 배열에 강함)
    // =========================================================
    static int minLenSubarraySumAtLeastK(int[] arr, long k) {
        long sum = 0;
        int left = 0;
        int ans = Integer.MAX_VALUE;
        for (int right = 0; right < arr.length; right++) {
            sum += arr[right];
            while (sum >= k) {
                ans = Math.min(ans, right - left + 1);
                sum -= arr[left++];
            }
        }
        return ans == Integer.MAX_VALUE ? 0 : ans;
    }

    // =========================================================
    // 9) Prefix Sum + HashMap (부분배열 합이 K인 개수)
    // =========================================================
    static long countSubarraysSumK(int[] arr, long k) {
        long ans = 0;
        long pref = 0;
        HashMap<Long, Integer> cnt = new HashMap<>();
        cnt.put(0L, 1);
        for (int x : arr) {
            pref += x;
            ans += cnt.getOrDefault(pref - k, 0);
            cnt.put(pref, cnt.getOrDefault(pref, 0) + 1);
        }
        return ans;
    }

    // =========================================================
    // 10) Binary Search (Parametric): 최소 True 찾기
    // =========================================================
    interface Check {
        boolean ok(long x);
    }
    static long parametricSearchMinTrue(long lo, long hi, Check check) {
        while (lo < hi) {
            long mid = (lo + hi) >>> 1;
            if (check.ok(mid)) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }

    // =========================================================
    // 11) Heap: Top-K largest (min-heap size K 유지)
    // =========================================================
    static int[] topKLargest(int[] nums, int k) {
        if (k <= 0) return new int[0];
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int x : nums) {
            if (pq.size() < k) pq.add(x);
            else if (x > pq.peek()) {
                pq.poll();
                pq.add(x);
            }
        }
        int[] out = new int[pq.size()];
        for (int i = out.length - 1; i >= 0; i--) out[i] = pq.poll();
        // reverse to descending
        for (int i = 0, j = out.length - 1; i < j; i++, j--) {
            int tmp = out[i]; out[i] = out[j]; out[j] = tmp;
        }
        return out;
    }

    // =========================================================
    // 12) Fenwick Tree (BIT) - 구간합 + 업데이트
    // =========================================================
    static class Fenwick {
        int n;
        long[] bit;
        Fenwick(int n) { this.n = n; bit = new long[n + 1]; }

        void add(int idx0, long delta) {
            for (int i = idx0 + 1; i <= n; i += i & -i) bit[i] += delta;
        }

        long sumPrefix(int idx0) { // [0..idx0]
            long s = 0;
            for (int i = idx0 + 1; i > 0; i -= i & -i) s += bit[i];
            return s;
        }

        long sumRange(int l0, int r0) { // [l0..r0]
            if (l0 > r0) return 0;
            return sumPrefix(r0) - (l0 > 0 ? sumPrefix(l0 - 1) : 0);
        }
    }

    // =========================================================
    // 13) Segment Tree (range min) - 구간최솟값 + 업데이트
    //     필요하면 min -> max / sum 으로 merge만 바꿔서 사용
    // =========================================================
    static class SegTreeMin {
        int n;
        long[] t;
        final long DEFAULT = INF;

        SegTreeMin(long[] arr) {
            int m = arr.length;
            n = 1;
            while (n < m) n <<= 1;
            t = new long[n << 1];
            Arrays.fill(t, DEFAULT);
            for (int i = 0; i < m; i++) t[n + i] = arr[i];
            for (int i = n - 1; i >= 1; i--) t[i] = Math.min(t[i << 1], t[i << 1 | 1]);
        }

        void update(int idx, long value) {
            int i = n + idx;
            t[i] = value;
            for (i >>= 1; i >= 1; i >>= 1) t[i] = Math.min(t[i << 1], t[i << 1 | 1]);
        }

        long query(int l, int rExclusive) {
            long res = DEFAULT;
            int L = n + l, R = n + rExclusive;
            while (L < R) {
                if ((L & 1) == 1) res = Math.min(res, t[L++]);
                if ((R & 1) == 1) res = Math.min(res, t[--R]);
                L >>= 1; R >>= 1;
            }
            return res;
        }
    }

    // =========================================================
    // 14) DSU (Union-Find) - 연결/그룹
    // =========================================================
    static class DSU {
        int[] p, sz;
        DSU(int n) {
            p = new int[n];
            sz = new int[n];
            for (int i = 0; i < n; i++) { p[i] = i; sz[i] = 1; }
        }
        int find(int x) {
            while (p[x] != x) {
                p[x] = p[p[x]];
                x = p[x];
            }
            return x;
        }
        boolean union(int a, int b) {
            int ra = find(a), rb = find(b);
            if (ra == rb) return false;
            if (sz[ra] < sz[rb]) { int t = ra; ra = rb; rb = t; }
            p[rb] = ra;
            sz[ra] += sz[rb];
            return true;
        }
    }

    public static void main(String[] args) {
        // toolkit file: usually not executed directly
    }
}