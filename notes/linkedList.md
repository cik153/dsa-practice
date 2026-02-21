# LinkedList

## What it is
**ArrayList**는 데이터를 “노드(node)” 단위로 저장하고, 각 노드가 다음 노드를 가리키는(링크하는) 자료구조
- 새로운 노드를 삽입시에, 그 삽입되는 곳의 앞 노드의 link에 저장된 주소를 새로운 노드의 link에 넣어 삽입될 곳의 뒷 노드를 연결시켜주고, 삽입되는 곳의 앞 노드의 link에 새로운 노드의 주소값을 넣어 연결시켜준다.
- newNode.link <- preNode.link
- preNode.link <- newNode