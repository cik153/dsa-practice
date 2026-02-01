package MyTrajectoryNum;

public class MyTrajectoryNum {
	public static void main(String[] args) {
		test("UULLDUDUDUD");
	}
	
	public static class Point {
	    int row = 0;
	    int column = 0;
	}
	
    public static int test (String dirs){
    	Point[] point = new Point[500];
    	for (int i = 0; i < point.length; i++) {
    	    point[i] = new Point();
    	}
    	
        int count = 0;
        int result = 0;
        
        for(int index = 0; index < dirs.length(); index++) {
            int row = point[count].row;
            int column = point[count].column;
            switch(dirs.charAt(index)){
                case 'U':
                	row++;
                    break;
                case 'D':
                	row--;
                    break;
                case 'R':
                    column++;
                    break;
                case 'L':
                	column--;
                    break;
            }
            
            if (row > 5 || row < -5 || column > 5 || column < -5) continue;
            
            count++;
            point[count].row = row;
            point[count].column = column;
            if(!CheckDuplication(count, row, column, point)){
                result++;
            }
        }
        return result;
    }
    
    public static boolean CheckDuplication(int pointsIndex, int row, int column, Point[] points){
        for(int index = 0; index < pointsIndex; index++){
            if(points[index].row == row && points[index].column == column){
                if(index > 0){
                    if(points[index-1].row == points[pointsIndex-1].row && points[index-1].column == points[pointsIndex-1].column){
                        return true;
                    }
                }
                if(points[index+1].row == points[pointsIndex-1].row && points[index+1].column == points[pointsIndex-1].column){
                    return true;
                }
            }
        }
        return false;
    }
}