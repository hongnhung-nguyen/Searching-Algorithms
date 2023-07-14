import os
import matplotlib.pyplot as plt
from audioop import reverse
from heapq import heappop
from heapq import heappush
import turtle
import pyautogui
from collections import deque as queue
from queue import PriorityQueue
from PIL import Image
import tkinter
import math
import sys
window = turtle.Screen()
window.screensize(1000,1000)
turtle.delay(0)
turtle.hideturtle()
turtle.penup()
turtle.speed(0)
os.makedirs(r'output\level_1\input1\dfs')
os.makedirs(r'output\level_1\input2\dfs')
os.makedirs(r'output\level_1\input3\dfs')
os.makedirs(r'output\level_1\input4\dfs')
os.makedirs(r'output\level_1\input5\dfs')

os.makedirs(r'output\level_1\input1\bfs')
os.makedirs(r'output\level_1\input2\bfs')
os.makedirs(r'output\level_1\input3\bfs')
os.makedirs(r'output\level_1\input4\bfs')
os.makedirs(r'output\level_1\input5\bfs')

os.makedirs(r'output\level_1\input1\ucs')
os.makedirs(r'output\level_1\input2\ucs')
os.makedirs(r'output\level_1\input3\ucs')
os.makedirs(r'output\level_1\input4\ucs')
os.makedirs(r'output\level_1\input5\ucs')

os.makedirs(r'output\level_1\input1\gbfs')
os.makedirs(r'output\level_1\input2\gbfs')
os.makedirs(r'output\level_1\input3\gbfs')
os.makedirs(r'output\level_1\input4\gbfs')
os.makedirs(r'output\level_1\input5\gbfs')

os.makedirs(r'output\level_1\input1\astar')
os.makedirs(r'output\level_1\input2\astar')
os.makedirs(r'output\level_1\input3\astar')
os.makedirs(r'output\level_1\input4\astar')
os.makedirs(r'output\level_1\input5\astar')

os.makedirs(r'output\level_2')
os.makedirs(r'output\advance')

#os.makedirs('output\level_1\DFS')
#os.makedirs('Output\level_1\BFS')
#os.makedirs(r'Output\level_1\UCS')
#os.makedirs('Output\level_1\ASearch')
#os.makedirs('Output\level_1\GBFS')
#os.makedirs('Output\level_2')
#os.makedirs(r'Output\advance')

def drawBox(x, y, Lcolor, Tcolor):
    turtle.goto(x, y)
    turtle.pendown()
    turtle.color(Lcolor, Tcolor)
    turtle.begin_fill()
    for i in range(4):  
        turtle.fd(20)
        turtle.right(90)
    turtle.end_fill()
    turtle.penup()

def drawGrid(ROW,COL):
    M = -330
    N = -230
    count = 20
    for i in range(COL):
        for j in range(ROW):
            drawBox(M, N, 'black', 'white')
            M += count
        M = -330
        N += count

def drawObstacles(x, y, colorT):
    M = -350
    N = -250
    drawBox(M + x * 20, N + y * 20, 'black', colorT)  
def CanMove(x1, y1, ROW,COL,matrix):
    if(x1 < 1 or y1 < 1  or
        x1 >= ROW + 1 or y1 >= COL + 1):
        return False
    elif (matrix[x1-1][y1-1]=='x'):
        return False
    else:
        return True;
def ColorPath(path):
    for i in range(1, len(path) - 1):
        t = path[i]
        drawObstacles(t[0],t[1],'blue')

def ManhattanDis(x1,y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)  
def Distance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
def filePATH(pre: str):
    def realPATH(fileName: str):
        return pre+fileName
    return realPATH

def ASearch1(x1, y1, x2, y2, filename,ROW,COL,matrix, namepath):  
    drawGrid(ROW,COL)
    for i in range(ROW ):
       for j in range( COL ):
            if matrix[i][j]=='x':
                drawObstacles(i+1,j+1,'black')
    drawObstacles(x1,y1, 'red') 
    global PATHCOST
    frontier = PriorityQueue()
    frontier.put((ManhattanDis(x1,y1,x2,y2),x1,y1,[]))
    thePath=filePATH('output\\level_1\\'  + namepath+ '\\astar\\')
    while(frontier.qsize() > 0):   
        temp = frontier.get()
        a = temp[1]
        b = temp[2]     
        matrix[a-1][b-1]='x'
        temp[3].append((a, b))
        if(a == x2 and b == y2):
            with open(thePath('astar_heuristic_1.txt'),'w') as fp:
                fp.write('THE PATH COST IS: ')
                fp.write(str(len(temp[3])))
                fp.close()
            drawObstacles(x2, y2, 'red')
            ColorPath(temp[3])
            #print (len(temp))
            return temp[3]
        #Go heading
        Dis1 = 100
        Dis2 = 100
        Dis3 = 100
        Dis4 = 100
        if(CanMove(a, b + 1,ROW,COL,matrix)):
                PATHCOST += 1
                if(matrix[a-1][(b + 1)-1] != 'x'):
                    matrix[a-1][(b + 1)-1] = 'x'
                    drawObstacles(a, b + 1, 'orange')
                    frontier.put((ManhattanDis(a,b + 1, x2, y2)+ len(temp[3]),a , b + 1, temp[3][:], ))
        #Go down
        if(CanMove(a, b - 1,ROW,COL,matrix)):
                PATHCOST += 1
                if(matrix[a-1][(b - 1)-1] != 'x'):
                    matrix[a-1][(b - 1)-1] = 'x'
                    frontier.put((ManhattanDis(a,b - 1, x2, y2)+ len(temp[3]),a , b - 1, temp[3][:]))
                    drawObstacles(a, b - 1, 'orange')
        if(CanMove(a - 1, b,ROW,COL,matrix)):
                PATHCOST += 1
                if(matrix[(a - 1)-1][b-1] != 'x'):
                    matrix[(a - 1)-1][b-1] = 'x'
                    frontier.put((ManhattanDis(a - 1,b, x2, y2) + len(temp[3]),a -1, b, temp[3][:]))
                    drawObstacles(a - 1, b, 'orange')
        if(CanMove(a + 1, b,ROW,COL,matrix)):
                PATHCOST += 1
                if(matrix[a][b-1] != 'x'):
                    matrix[a][b-1] = 'x'
                    frontier.put((ManhattanDis(a+1,b, x2, y2) + len(temp[3]),a + 1 , b, temp[3][:]))
                    drawObstacles(a + 1, b, 'orange')
    with open(thePath('astar_heuristic_1.txt'),'w+') as fp:
        fp.write('NO TARGET IS NOT FOUND ')  
        fp.close()
def ASearch2(x1, y1, x2, y2, filename,ROW,COL,matrix, namepath):  
    drawGrid(ROW,COL)
    for i in range(ROW ):
       for j in range( COL ):
            if matrix[i][j]=='x':
                drawObstacles(i+1,j+1,'black')
    drawObstacles(x1,y1, 'red') 
    global PATHCOST
    frontier2 = PriorityQueue()
    frontier2.put((Distance(x1,y1,x2,y2),x1,y1,[]))
    thePath=filePATH('output\\level_1\\' + namepath +'\\astar\\')
   
    while(frontier2.qsize() > 0):   
        temp = frontier2.get()
        a = temp[1]
        b = temp[2]     
        matrix[a-1][b-1]='x'
        temp[3].append((a, b))
        if(a == x2 and b == y2):
            with open(thePath('astar_heuristic_2.txt'),'w') as fp:
                fp.write('THE PATH COST IS: ')
                fp.write(str(len(temp[3])))
                fp.close()
            drawObstacles(x2, y2, 'red')
            ColorPath(temp[3])
            return temp[3]
        #Go heading
        Dis1 = 100
        Dis2 = 100
        Dis3 = 100
        Dis4 = 100
        if(CanMove(a, b + 1,ROW,COL,matrix)):
                PATHCOST += 1
                if(matrix[a-1][(b + 1)-1] != 'x'):
                    matrix[a-1][(b + 1)-1] = 'x'
                    drawObstacles(a, b + 1, 'orange')
                    frontier2.put((Distance(a,b + 1, x2, y2)+ len(temp[3]),a , b + 1, temp[3][:], ))
        #Go down
        if(CanMove(a, b - 1,ROW,COL,matrix)):
                PATHCOST += 1
                if(matrix[a-1][(b - 1)-1] != 'x'):
                    matrix[a-1][(b - 1)-1] = 'x'
                    frontier2.put((Distance(a,b - 1, x2, y2)+ len(temp[3]),a , b - 1, temp[3][:]))
                    drawObstacles(a, b - 1, 'orange')
        if(CanMove(a - 1, b,ROW,COL,matrix)):
                PATHCOST += 1
                if(matrix[(a - 1)-1][b-1] != 'x'):
                    matrix[(a - 1)-1][b-1] = 'x'
                    frontier2.put((Distance(a - 1,b, x2, y2) + len(temp[3]),a -1, b, temp[3][:]))
                    drawObstacles(a - 1, b, 'orange')
        if(CanMove(a + 1, b,ROW,COL,matrix)):
                PATHCOST += 1
                if(matrix[a][b-1] != 'x'):
                    matrix[a][b-1] = 'x'
                    frontier2.put((Distance(a+1,b, x2, y2) + len(temp[3]),a + 1 , b, temp[3][:]))
                    drawObstacles(a + 1, b, 'orange')
    with open(thePath('astar_heuristic_2.txt'),'w+') as fp:
        fp.write('NO TARGET IS NOT FOUND ')  
        fp.close()
def UCS(x1, y1, x2, y2,filename,ROW,COL,matrix, namepath):
    drawGrid(ROW,COL)
    for i in range(ROW ):
       for j in range( COL ):
            if matrix[i][j]=='x':
                drawObstacles(i+1,j+1,'black')
    drawObstacles(x1,y1, 'red')
    thePath=filePATH('output\\level_1\\'+ namepath+'\\ucs\\')
    global PATHCOST
    PATHCOST=0
    queue = PriorityQueue()
    queue.put((0, x1, y1, []))   
    while queue.qsize() != 0:
        cost, a, b, temp = queue.get()
        temp.append((a,b))
        if(CanMove(a, b,ROW,COL,matrix)):
            matrix[a-1][b-1]='x'
            if a == x2 and b == y2:
                with open(thePath('ucs.txt'),'w') as fp:
                    fp.write('THE PATH COST IS: ')
                    fp.write(str(len(temp)))
                    fp.close()
                drawObstacles(x2, y2, 'red')
                ColorPath(temp)
                #print (len(temp))
                return 
            if CanMove(a, b + 1,ROW,COL,matrix):
                PATHCOST += 1
                drawObstacles(a, b + 1, 'yellow') 
                total_cost = cost + 1
                queue.put((total_cost, a, b + 1, temp[:]))                
            if CanMove(a, b - 1,ROW,COL,matrix):
                PATHCOST += 1
                drawObstacles(a, b - 1, 'yellow')
                total_cost = cost + 1
                queue.put((total_cost, a, b - 1, temp[:]))
            if CanMove(a + 1, b,ROW,COL,matrix):
                PATHCOST += 1 
                drawObstacles(a + 1, b, 'yellow')
                total_cost = cost + 1
                queue.put((total_cost, a + 1, b, temp[:]))
            if CanMove(a - 1 ,b,ROW,COL,matrix):
                PATHCOST += 1
                drawObstacles(a - 1, b, 'yellow')
                total_cost = cost + 1
                queue.put((total_cost, a - 1, b, temp[:]))
    if(queue.empty()):
        with open(thePath('ucs.txt'),'w+') as fp:
            fp.write('NO TARGET IS NOT FOUND ')  
            fp.close()
def rotated(array_2d):
    list_of_tuples = zip(*array_2d[::-1])
    return [list(elem) for elem in list_of_tuples]

def visualize_maze(matrix, bonus, start, end, route=None):
    """
    Args:
      1. matrix: The matrix read from the input file,
      2. bonus: The array of bonus points,
      3. start, end: The starting and ending points,
      4. route: The route from the starting point to the ending one, defined by an array of (x, y), e.g. route = [(1, 2), (1, 3), (1, 4)]
    """
    #1. Define walls and array of direction based on the route
    walls=[(i,j) for i in range(len(matrix)) for j in range(len(matrix[0])) if matrix[i][j]=='x']

    if route:
        direction=[]
        for i in range(1,len(route)):
            if route[i][0]-route[i-1][0]>0:
                direction.append('v') #^
            elif route[i][0]-route[i-1][0]<0:
                direction.append('^') #v        
            elif route[i][1]-route[i-1][1]>0:
                direction.append('>')
            else:
                direction.append('<')

        direction.pop(0)

    #2. Drawing the map
    ax=plt.figure(dpi=100).add_subplot(111)

    for i in ['top','bottom','right','left']:
        ax.spines[i].set_visible(False)

    plt.scatter([i[1] for i in walls],[-i[0] for i in walls],
                marker='X',s=100,color='black')
    
    plt.scatter([i[1] for i in bonus],[-i[0] for i in bonus],
                marker='P',s=100,color='green')

    plt.scatter(start[1],-start[0],marker='*',
                s=100,color='gold')

    if route:
        for i in range(len(route)-2):
            plt.scatter(route[i+1][1],-route[i+1][0],
                        marker=direction[i],color='silver')

    plt.text(end[1],-end[0],'EXIT',color='red',
        horizontalalignment='center',
        verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
    



def bfs(graph, start, end):
    queue=[]
    tracking={start:(0,0)}
    path=[]
    current=start

    while current!=end:     
        for neighbor in graph[current]:     # Xét 4 vị trí liền kề vị trí đang xét current
            if neighbor not in tracking:    # Nếu vị trí neighbor chưa được xét trước đó
                queue.append(neighbor)      # Thêm vị trí neighbor vào queue
                tracking[neighbor]=current  # Đánh dấu vị trí cha đi đến neighbor là current
        current=queue.pop(0)
    
    # Backtracking
    temp=end
    while temp!=(0,0):
        path.insert(0,tracking[temp])
        temp=tracking[temp]
    path.append(end)
    return path[1:]


def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def find_path(graph, begin, end, past_point, bonus_points):
    heap = [(0, begin)]
    info = {begin:[0,heuristic(begin, end),0,(0,0)]}
    way_out = False     # Biến cho biết có đường đi từ begin đến end hay không

    while (len(heap)):      # Lặp lại khi Vẫn còn biên chưa mở
        current = heappop(heap)[1]      # Mở biên có f = g + h + b là nhỏ nhất, current =  vị trí (x,y) tại đó
        
        if current==end:
            way_out = True      # Tìm được đến end -> có đường đi
            break             

        for neighbor in graph[current]:
            if neighbor not in info:
                g = info[current][0] + 1
                h = heuristic(neighbor, end)
                b = 0       # Mặc định các vị trí đều có điểm thưởng b = 0
                if neighbor not in past_point:      # Nếu đó là vị trí điểm thưởng
                    for pts in bonus_points:        # và chưa được ăn (không có trong đoạn đường start->begin đã đi qua)
                        if pts[0]==neighbor[0] and pts[1]==neighbor[1]:
                            b = pts[2]              # b = giá trị điểm thưởng lấy từ list bonus_points
                            break
                info[neighbor] = [g, h, b, current]
                heappush(heap,(g+h+b, neighbor))
            elif info[neighbor][0] > info[current][0] + 1:
                info[neighbor][0] = info[current][0] + 1
                info[neighbor][3] = current

    path = []
    cost = 0
    track = end
    # Nếu có đường đi thì backtracking tìm path và cost
    if way_out:
        while track!=(0,0):
            path.insert(0, track)
            cost += 1 + info[track][2]
            track = info[track][3]
        cost -= 1       # Dư 1 bước do đã ở sẵn vị trí begin chứ không phải bước đến từ (0,0)
    return [way_out, cost, path]

# Tìm đường đi từ start -> end mà tốn ít chi phí nhất trên bản đồ mê cung có điểm thưởng
def solve_bonus_map(graph, start, end, rows, cols, bonus_points):
    queue = [start]

    # short_matrix là dictionary lưu các key là start, end và các bonus point
    # value gồm [0]: chi phí đi từ start đến key, [1]: đường đi đến key
    short_matrix = {(x[0],x[1]):[rows*cols, []] for x in bonus_points}
    short_matrix[start] = [0, [start]]
    short_matrix[end] = [rows*cols, []]

    while len(queue):
        current = queue.pop(0)
        for point in short_matrix:
            # Chỉ xét các điểm không nằm trong đoạn đường start->current đã đi qua
            if point not in short_matrix[current][1]:
                wayout = find_path(graph,current, point, short_matrix[current][1],bonus_points)
                
                if not wayout[0]:       # Nếu không tìm được đường đi current->point thì bỏ qua
                    continue

                # Nếu đường đi cũ lớn hơn đường đi mới thông qua current thì update đường đi nhỏ hơn
                # (start -> point) > (start -> current -> point)
                if short_matrix[point][0] >= short_matrix[current][0] + wayout[1]:
                    short_matrix[point][0] = short_matrix[current][0] + wayout[1]
                    short_matrix[point][1] = short_matrix[current][1] + wayout[2][1:]
                    # Vì point vừa được update nên thêm vào queue để xét đường đi mới đến các điểm khác
                    if point!=end and point not in queue:
                        queue.append(point)
    return short_matrix[end]

def read_filelv1(file_name):
  f=open(os.path.join('Input\level_1', file_name), 'r')
  n_bonus_points = int(next(f)[:-1])
  bonus_points = []
  for i in range(n_bonus_points):
    x, y, reward = map(int, next(f)[:-1].split(' '))
    bonus_points.append((x, y, reward))

  text=f.read()
  matrix=[list(i) for i in text.splitlines()]
  f.close()

  return bonus_points, matrix

def read_filelv2(file_name):
  f=open(os.path.join('Input\level_2', file_name), 'r')
  n_bonus_points = int(next(f)[:-1])
  bonus_points = []
  for i in range(n_bonus_points):
    x, y, reward = map(int, next(f)[:-1].split(' '))
    bonus_points.append((x, y, reward))

  text=f.read()
  matrix=[list(i) for i in text.splitlines()]
  f.close()

  return bonus_points, matrix

def main():
    


    for filename in os.listdir('Input\level_1'):
        bonus_points, matrix = read_filelv1(filename)
        # Xác định 2 điểm đầu cuối
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j]=='S':
                    start=(i,j)

                elif matrix[i][j]==' ':
                    if (i==0) or (i==len(matrix)-1) or (j==0) or (j==len(matrix[0])-1):
                        end=(i,j)
                    
                else:
                    pass
                
        # Graph lưu vị trí liền kề có thể đi tới được từ vị trí (i,j)
        rows=len(matrix)
        cols=len(matrix[0])
        graph={}
        for i in range(1,rows-1):
            for j in range(1,cols-1):
                if matrix[i][j]!='x':
                    adj=[]
                    for loc in [(i,j+1),(i,j-1),(i+1,j),(i-1,j)]:
                        if matrix[loc[0]][loc[1]]!='x':
                            adj.append(loc)
                    graph[(i,j)]=adj
        if end[0]==0:
            graph[end] = [(end[0]+1,end[1])]
        elif end[0]==rows-1:
            graph[end] = [(end[0]-1,end[1])]
        elif end[1]==0:
            graph[end] = [(end[0],end[1]+1)]
        else:
            graph[end] = [(end[0],end[1]-1)]
        filename= filename[0: (len(filename)-4)]            #
        save_path='output/level_1/'+filename+'/bfs/'
        complete_name=os.path.join(save_path,'bfs.txt' )
        f = open(complete_name, 'w')
        sys.stdout = f
        wayoutBFS = bfs(graph, start, end)
        #print(road)
        print(len(wayoutBFS)-1)

        visualize_maze(matrix,bonus_points,start,end,wayoutBFS)
        plt.savefig('./output/level_1/'+filename+'/bfs/bfs.jpg')
        #plt.show()
        f.close()

    for filename in os.listdir('Input\level_2'):
        bonus_points, matrix = read_filelv2(filename)
        # Xác định 2 điểm đầu cuối
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j]=='S':
                    start=(i,j)

                elif matrix[i][j]==' ':
                    if (i==0) or (i==len(matrix)-1) or (j==0) or (j==len(matrix[0])-1):
                        end=(i,j)
                    
                else:
                    pass
                
        # Graph lưu vị trí liền kề có thể đi tới được từ vị trí (i,j)
        rows=len(matrix)
        cols=len(matrix[0])
        graph={}
        for i in range(1,rows-1):
            for j in range(1,cols-1):
                if matrix[i][j]!='x':
                    adj=[]
                    for loc in [(i,j+1),(i,j-1),(i+1,j),(i-1,j)]:
                        if matrix[loc[0]][loc[1]]!='x':
                            adj.append(loc)
                    graph[(i,j)]=adj
        if end[0]==0:
            graph[end] = [(end[0]+1,end[1])]
        elif end[0]==rows-1:
            graph[end] = [(end[0]-1,end[1])]
        elif end[1]==0:
            graph[end] = [(end[0],end[1]+1)]
        else:
            graph[end] = [(end[0],end[1]-1)]
        filename= filename[0: (len(filename)-4)]         
        save_path='output/level_2'
        complete_name=os.path.join(save_path,filename+".txt" )
        f = open(complete_name, 'w')
        sys.stdout = f
        #wayoutBFS = bfs(graph, start, end)
        #print(road)
        ans = solve_bonus_map(graph, start, end,rows,cols,bonus_points)
        print(f'Cost = {ans[0]}')
        visualize_maze(matrix,bonus_points,start,end,ans[1])
        plt.savefig('./output/level_2/'+filename+'.jpg')
        #plt.show()
        f.close()


############# GBFS
    frontier=[]
    explored=[]
    direction= [(-1,0) , (0,-1), (1,0) , (0,1)  ]
    parent={}

    def is_in_matrix(matrix, node):
        if node[0]<0 or node[1]<0:
            return False
        if node[0]> len(matrix)-1 and node[1]> len(matrix[0])-1:
            return False
        return True

    def set_parent(parent_node, child_node):
        tg={child_node: parent_node}
        parent.update(tg)
        return child_node
    # kiem tra lai phan nay neu code sai
    def add_frontier_GBFS(matrix, current, end ):
        for i in range(0,4):
            x= current[0] + direction[i][0]
            y= current[1] + direction[i][1]
            node= (x,y)
            if node== end:
                set_parent(current, end)
                return 0
            if (node not in frontier) and is_in_matrix(matrix, node) and node not in explored and (matrix[node[0]][node[1]] !='x'):
                frontier.append(set_parent(current, node))
        return 1

    def heuristic_1(node):
        dx = abs(node[0] - end[0]) 
        dy = abs(node[1] - end[1]) 
        return (dx + dy)
    def heuristic_2(node):
        dx = abs(node[0] - end[0]) 
        dy = abs(node[1] - end[1]) 
        return math.sqrt(dx * dx + dy * dy)

    def sort_frontier(sort_by):
        frontier.sort(key=sort_by)

    def Greedy_breath_first_search(matrix,bonus_points, start, end, Heuristic):
        frontier.clear()
        explored.clear()
        parent.clear() 
        path =[]
        current=start
        cost=0
    # kiem tra co phai la diem end 
        if current ==end:
            return path
        while (add_frontier_GBFS(matrix, current, end)):
            if len(frontier)==0:
                return path, -1
            explored.append(current)
            sort_frontier(Heuristic)
            current= frontier.pop(0)
        explored.append(current)
        current=end
        while True:
            path.insert(0, current)
            if current== start:
                break
            current= parent[current]
            cost += 1
        return path, cost
    for filename in os.listdir('Input\level_1'):
        bonus_points, matrix = read_filelv1(filename)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j]=='S':
                    start=(i,j)

                elif matrix[i][j]==' ':
                    if (i==0) or (i==len(matrix)-1) or (j==0) or (j==len(matrix[0])-1):
                        end=(i,j)

                else:
                    pass
        filename= filename[0: (len(filename)-4)]       
        savepath='output/level_1/'+filename + '/gbfs'
        complete_name=os.path.join(savepath,"gbfs_heuristic_1.txt" )
        f = open(complete_name, 'w')
        sys.stdout = f
        path, cost=Greedy_breath_first_search(matrix,bonus_points, start, end, heuristic_1)
        print(cost)
        visualize_maze(matrix,bonus_points,start,end, path )
        plt.savefig('./output/level_1/'+filename+'/gbfs/gbfs_heuristic_1.jpg')
        #plt.show()
        f.close()
        complete_name=os.path.join(savepath,"gbfs_heuristic_2.txt" )
        f = open(complete_name, 'w')
        sys.stdout = f
        path, cost=Greedy_breath_first_search(matrix,bonus_points, start, end, heuristic_2)
        print(cost)
        visualize_maze(matrix,bonus_points,start,end, path )
        plt.savefig('./output/level_1/'+filename+'/gbfs/gbfs_heuristic_2.jpg')
        #plt.show()
        f.close()

    
    def add_frontier_DFS(matrix, current, end ):
        for i in range(0,4):
            x= current[0] + direction[i][0]
            y= current[1] + direction[i][1]
            node= (x,y)
            #print(node)
            if node== end:
                set_parent(current, end)
                return 0
            if (node not in frontier) and is_in_matrix(matrix, node) and node not in explored and (matrix[node[0]][node[1]] !='x'):
                frontier.insert(0, set_parent( current, node))
        return 1


    def depth_first_search(matrix,bonus_points, start, end):
        frontier.clear()
        explored.clear()
        parent.clear() 
        road =[]
        current=start
        cost=0
        # kiem tra co phai la diem end 
        if current ==end:
            return road
        while (add_frontier_DFS(matrix, current, end)):
            if len(frontier)==0:
                return road, -1
            explored.append(current)
            current= frontier.pop(0)
        explored.append(current)
        current=end
        while True:
            road.insert(0, current)
      #  yellow.goto(current[0], current[1])                      
       # yellow.stamp()
            if current== start:
                break
            current= parent[current]
            cost += 1
        return road, cost

    for filename in os.listdir('Input\level_1'):
        bonus_points, matrix = read_filelv1(filename)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j]=='S':
                    start=(i,j)

                elif matrix[i][j]==' ':
                    if (i==0) or (i==len(matrix)-1) or (j==0) or (j==len(matrix[0])-1):
                        end=(i,j)
                
                else:
                    pass
        filename= filename[0: (len(filename)-4)]   
        savepath='output/level_1/' + filename + '/dfs'
        complete_name=os.path.join(savepath,"dfs.txt")
        f = open(complete_name, 'w')
        sys.stdout = f
        road, cost=depth_first_search(matrix,bonus_points, start, end)
        #print(road)
        if cost==-1:
            print("NO")
        else:
            print(cost)
        visualize_maze(matrix,bonus_points,start,end, road)
        plt.savefig('./output/level_1/'+filename+'/dfs/dfs.jpg')
        #plt.show()
        f.close()

    for filename in os.listdir('Input\level_1'):
        bonus_points, matrix = read_filelv1(filename)
        matrix = rotated(matrix)
        COL = 0 # columm of the maze
        ROW = 0 # row of the maze
        start = () # the starting point
        end = () # the end point
        ROW = len(matrix)
        COL = len(matrix[0])
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j]=='S':
                    start=(i,j)
                elif matrix[i][j]==' ':
                    if (i==0) or (i==len(matrix)-1) or (j==0) or (j==len(matrix[0])-1):
                        end=(i,j)
                else:
                    pass
        
        #PATHCOST=0
        filename= filename[0: (len(filename)-4)]
        UCS(start[0]+1,start[1]+1,end[0]+1,end[1]+1,filename,ROW,COL,matrix,filename )
        myScreenshot=pyautogui.screenshot()
      
        myScreenshot.save('./output/level_1/'+filename+'/ucs/ucs.jpg')
        turtle.clear()

    for filename in os.listdir('Input\level_1'):
        bonus_points, matrix = read_filelv1(filename)
        matrix = rotated(matrix)
        COL = 0 # columm of the maze
        ROW = 0 # row of the maze
        start = () # the starting point
        end = () # the end point
        ROW = len(matrix)
        COL = len(matrix[0])
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j]=='S':
                    start=(i,j)
                elif matrix[i][j]==' ':
                    if (i==0) or (i==len(matrix)-1) or (j==0) or (j==len(matrix[0])-1):
                        end=(i,j)
                else:
                    pass
        #PATHCOST = 0
        filename= filename[0: (len(filename)-4)]

        ASearch1(start[0]+1,start[1]+1,end[0]+1,end[1]+1,filename,ROW,COL,matrix, filename)
        myScreenshot=pyautogui.screenshot()
        
        myScreenshot.save('./output/level_1/'+filename+'/astar/astar_heuristic_1.jpg')
        turtle.clear()

    for filename in os.listdir('Input\level_1'):
        bonus_points, matrix = read_filelv1(filename)
        matrix = rotated(matrix)
        COL = 0 # columm of the maze
        ROW = 0 # row of the maze
        start = () # the starting point
        end = () # the end point
        ROW = len(matrix)
        COL = len(matrix[0])
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j]=='S':
                    start=(i,j)
                elif matrix[i][j]==' ':
                    if (i==0) or (i==len(matrix)-1) or (j==0) or (j==len(matrix[0])-1):
                        end=(i,j)
                else:
                    pass
        #PATHCOST = 0
        filename= filename[0: (len(filename)-4)]
        ASearch2(start[0]+1,start[1]+1,end[0]+1,end[1]+1,filename,ROW,COL,matrix, filename)
        myScreenshot=pyautogui.screenshot()
        myScreenshot.save('./output/level_1/'+filename+'/astar/astar_heuristic_2.jpg')
        turtle.clear()
    turtle.done()
    
if __name__=="__main__":
    main()