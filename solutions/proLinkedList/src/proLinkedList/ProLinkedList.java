package proLinkedList;
import java.util.List;
public class ProLinkedList {
	public static void main(String[] args) {
	}
	/*
	 * public class Node<E> { public E data; public Node<E> link;
	 * 
	 * public Node(E newData) { data = newData; link = null; }
	 * 
	 * public Node(E newData, Node<E> newNode) { data = newData; link = newNode; } }
	 */
	
	public class Node {
		public Object item;
		public Node next;
		
		public Node(Object newItem) {
			item = newItem;
			next = null;
		}
		
		public Node(Object newItem, Node nextNode) {
			item = newItem;
			next = nextNode;
		}
	}
	
	public class LinkedList<E> implements ListInterface<E> {
		private Node<E> head;
		private int numItems;

		public LinkedList() { 			
			numItems = 0;
			head = new Node<>(null, null); 	
		}

		
		public void add(int index, E item) {
			if (index >= 0 && index <= numItems) {
				Node<E> prevNode = getNode(index - 1);
				Node<E> newNode = new Node<>(item, prevNode.next);
				prevNode.next = newNode;
				numItems++;
			}
		}
	 
		public void append(E item) {
			Node<E> prevNode = head;  		
			while (prevNode.next != null) {
				prevNode = prevNode.next;
			}
			Node<E> newNode = new Node<>(item, null);
			prevNode.next = newNode;
			numItems++;
		}

		public E remove(int index) {
			if (index >= 0 && index < numItems) {
				Node<E> prevNode = getNode(index - 1);
				Node<E> currNode = prevNode.next;
				prevNode.next = currNode.next;
				numItems--;
				return currNode.item;
			} else
				return null;
		}

		public boolean removeItem(E x) {
			Node<E> currNode = head; 
			Node<E> prevNode;
			for (int i = 0; i < numItems; i++) {
				prevNode = currNode;
				currNode = currNode.next;
				if (((Comparable)(currNode.item)).compareTo(x) == 0) {
					prevNode.next = currNode.next;
					numItems--;
					return true;
				}
			}
			return false; 			
		}

		public E get(int index) {
			if (index >= 0 && index <= numItems - 1) {
				return getNode(index).item;
			} else
				return null; 		
		}
	 
			public Node<E> getNode(int index) {
			if (index >= -1 && index <= numItems - 1) {
				Node<E> currNode = head;  	
				for (int i = 0; i <= index; i++) {
					currNode = currNode.next;
				}
				return currNode;
			} else {
				return null;
			}
		}

		public void set(int index, E x) {
			if (index >= 0 && index <= numItems - 1) {
				getNode(index).item = x;
			} else { /* 에러 처리 코드작성 */ }
		}

		public final int NOT_FOUND = -12345;
		public int indexOf(E x) {
			Node<E> currNode = head;   
			int i;
			for (i = 0; i < numItems; i++) {
				currNode = currNode.next;
				if (((Comparable)(currNode.item)).compareTo(x) == 0)
					return i;
			}
			return NOT_FOUND; 	
		}
		
		public int len() {
			return numItems;
		}
		
		public boolean isEmpty() {
			return numItems == 0;
		}

		public void clear() {
			numItems = 0;
			head = new Node<>(null, null);
		}
		
		public void printAll() {
			Node<E> t;
			System.out.print("Print list (#items=" + numItems + ") ");
			for(t=head.next; t != null; t = t.next) {
				System.out.print(t.item + " ");
			}
			System.out.println();
		}
	} 

}

