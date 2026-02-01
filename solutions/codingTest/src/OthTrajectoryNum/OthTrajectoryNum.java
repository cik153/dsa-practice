package OthTrajectoryNum;

class OthTrajectoryNum {
	public static void main(String[] args) {
		solution("UULLDUDUDUD");
	}
	// 방향: 0=U, 1=D, 2=L, 3=R
    // 반대 방향: U<->D, L<->R
    private static final int[] OPP = {1, 0, 3, 2};

    public static int solution(String dirs) {
        // 좌표 -5..5 -> 인덱스 0..10
        boolean[][][] visited = new boolean[11][11][4];

        int x = 5, y = 5; // 시작 (0,0) -> (5,5)
        int result = 0;

        for (int i = 0; i < dirs.length(); i++) {
            int dir = toDir(dirs.charAt(i));
            if (dir == -1) continue;

            int nx = x, ny = y;
            if (dir == 0) ny++;      // U
            else if (dir == 1) ny--; // D
            else if (dir == 2) nx--; // L
            else nx++;               // R

            // 범위 밖이면 무시
            if (nx < 0 || nx > 10 || ny < 0 || ny > 10) continue;

            // (x,y) -> (nx,ny) 간선이 처음이면 카운트
            if (!visited[x][y][dir]) {
                visited[x][y][dir] = true;
                visited[nx][ny][OPP[dir]] = true; // 반대 방향도 방문 처리
                result++;
            }

            x = nx; y = ny;
        }

        return result;
    }

    private static int toDir(char c) {
        switch (c) {
            case 'U': return 0;
            case 'D': return 1;
            case 'L': return 2;
            case 'R': return 3;
            default:  return -1;
        }
    }
}
