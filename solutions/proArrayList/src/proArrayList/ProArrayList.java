package proArrayList;

import java.util.HashSet;

public class ProArrayList {
	public static void main(String[] args) {
		pro1Test();
	}
	
	public static void pro1Test() {
		int[] testData1 = {5, 4, 3, 5, 2, 10};
		int[] testData2 = {1, 165, 3, 5, 2, 10};
		int[] testData3 = {45, 24, 3, 5, 2, 10};
		
		System.out.println(pro1(testData1));
		System.out.println(pro1(testData2));
		System.out.println(pro1(testData3));
	}
	
	public static HashSet<Integer> pro1(int[] numList) {
		HashSet<Integer> set = new HashSet<>();
		for(int firstNum: numList) {
			for(int secondNum: numList) {
				set.add(firstNum + secondNum);
			}
		}
		return set;
	}
}