import sys
import timeit
from random import shuffle
from collections import deque
from heapq import heappush, heappop, heapify
from PySide6 import QtCore, QtGui, QtWidgets


class Ui_exp1(object):
    def setupUi(self, exp1):
        exp1.setObjectName("exp1")
        exp1.setEnabled(True)
        exp1.resize(600, 420)
        exp1.setMinimumSize(QtCore.QSize(600, 420))
        exp1.setMaximumSize(QtCore.QSize(600, 420))
        self.centralwidget = QtWidgets.QWidget(exp1)
        self.centralwidget.setObjectName("centralwidget")
        # self.img1 = QtOpenGLWidgets.QOpenGLWidget(self.centralwidget)
        # self.img1.setGeometry(QtCore.QRect(30, 30, 120, 120))
        # self.img1.setObjectName("img1")
        # self.img2 = QtOpenGLWidgets.QOpenGLWidget(self.centralwidget)
        # self.img2.setGeometry(QtCore.QRect(150, 30, 120, 120))
        # self.img2.setObjectName("img2")
        # self.img3 = QtOpenGLWidgets.QOpenGLWidget(self.centralwidget)
        # self.img3.setGeometry(QtCore.QRect(270, 30, 120, 120))
        # self.img3.setObjectName("img3")
        # self.img4 = QtOpenGLWidgets.QOpenGLWidget(self.centralwidget)
        # self.img4.setGeometry(QtCore.QRect(30, 150, 120, 120))
        # self.img4.setObjectName("img4")
        # self.img5 = QtOpenGLWidgets.QOpenGLWidget(self.centralwidget)
        # self.img5.setGeometry(QtCore.QRect(150, 150, 120, 120))
        # self.img5.setObjectName("img5")
        # self.img6 = QtOpenGLWidgets.QOpenGLWidget(self.centralwidget)
        # self.img6.setGeometry(QtCore.QRect(270, 150, 120, 120))
        # self.img6.setObjectName("img6")
        # self.img7 = QtOpenGLWidgets.QOpenGLWidget(self.centralwidget)
        # self.img7.setGeometry(QtCore.QRect(30, 270, 120, 120))
        # self.img7.setObjectName("img7")
        # self.img8 = QtOpenGLWidgets.QOpenGLWidget(self.centralwidget)
        # self.img8.setGeometry(QtCore.QRect(150, 270, 120, 120))
        # self.img8.setObjectName("img8")
        # self.img9 = QtOpenGLWidgets.QOpenGLWidget(self.centralwidget)
        # self.img9.setGeometry(QtCore.QRect(270, 270, 120, 120))
        # self.img9.setObjectName("img9")
        self.img1 = QtWidgets.QLabel(exp1)
        self.img1.setGeometry(QtCore.QRect(30, 30, 120, 120))
        self.img1.setObjectName("img1")
        self.img2 = QtWidgets.QLabel(exp1)
        self.img2.setGeometry(QtCore.QRect(150, 30, 120, 120))
        self.img2.setObjectName("img2")
        self.img3 = QtWidgets.QLabel(exp1)
        self.img3.setGeometry(QtCore.QRect(270, 30, 120, 120))
        self.img3.setObjectName("img3")
        self.img4 = QtWidgets.QLabel(exp1)
        self.img4.setGeometry(QtCore.QRect(30, 150, 120, 120))
        self.img4.setObjectName("img4")
        self.img5 = QtWidgets.QLabel(exp1)
        self.img5.setGeometry(QtCore.QRect(150, 150, 120, 120))
        self.img5.setObjectName("img5")
        self.img6 = QtWidgets.QLabel(exp1)
        self.img6.setGeometry(QtCore.QRect(270, 150, 120, 120))
        self.img6.setObjectName("img6")
        self.img7 = QtWidgets.QLabel(exp1)
        self.img7.setGeometry(QtCore.QRect(30, 270, 120, 120))
        self.img7.setObjectName("img7")
        self.img8 = QtWidgets.QLabel(exp1)
        self.img8.setGeometry(QtCore.QRect(150, 270, 120, 120))
        self.img8.setObjectName("img8")
        self.img9 = QtWidgets.QLabel(exp1)
        self.img9.setGeometry(QtCore.QRect(270, 270, 120, 120))
        self.img9.setObjectName("img9")
        self.startState = QtWidgets.QLineEdit(self.centralwidget)
        self.startState.setEnabled(False)
        self.startState.setGeometry(QtCore.QRect(420, 50, 149, 20))
        self.startState.setMaxLength(9)
        self.startState.setObjectName("startState")
        self.goalState = QtWidgets.QLineEdit(self.centralwidget)
        self.goalState.setGeometry(QtCore.QRect(420, 100, 149, 20))
        self.goalState.setMaxLength(9)
        self.goalState.setObjectName("goalState")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(420, 30, 71, 15))
        self.label.setObjectName("label")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(420, 125, 150, 265))
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_5 = QtWidgets.QWidget()
        self.page_5.setObjectName("page_5")
        self.shuffleButton = QtWidgets.QPushButton(self.page_5)
        self.shuffleButton.setGeometry(QtCore.QRect(0, 0, 149, 30))
        self.shuffleButton.setObjectName("shuffleButton")

        self.label_15 = QtWidgets.QLabel(self.page_5)
        self.label_15.setGeometry(QtCore.QRect(0, 45, 56, 15))
        self.label_15.setObjectName("label_15")
        self.byStep = QtWidgets.QPushButton(self.page_5)
        self.byStep.setEnabled(False)
        self.byStep.setGeometry(QtCore.QRect(0, 242, 149, 23))
        self.byStep.setObjectName("byStep")
        self.startSearch = QtWidgets.QPushButton(self.page_5)
        self.startSearch.setGeometry(QtCore.QRect(0, 118, 149, 91))
        self.startSearch.setObjectName("startSearch")
        self.puzzleButton1 = QtWidgets.QPushButton(self.page_5)
        self.puzzleButton1.setGeometry(QtCore.QRect(0, 214, 149, 23))
        self.puzzleButton1.setObjectName("puzzleButton1")
        self.bfs = QtWidgets.QRadioButton(self.page_5)
        self.bfs.setGeometry(QtCore.QRect(0, 65, 141, 20))
        self.bfs.setObjectName("bfs")
        self.astar = QtWidgets.QRadioButton(self.page_5)
        self.astar.setGeometry(QtCore.QRect(0, 90, 131, 20))
        self.astar.setChecked(True)
        self.astar.setObjectName("astar")
        self.stackedWidget.addWidget(self.page_5)
        self.page_6 = QtWidgets.QWidget()
        self.page_6.setObjectName("page_6")
        self.back = QtWidgets.QPushButton(self.page_6)
        self.back.setGeometry(QtCore.QRect(0, 242, 149, 23))
        self.back.setObjectName("back")
        self.preStep = QtWidgets.QPushButton(self.page_6)
        self.preStep.setGeometry(QtCore.QRect(0, 118, 74, 91))
        self.preStep.setObjectName("preStep")
        self.nextStep = QtWidgets.QPushButton(self.page_6)
        self.nextStep.setGeometry(QtCore.QRect(75, 118, 74, 91))
        self.nextStep.setObjectName("nextStep")
        self.puzzleGame2 = QtWidgets.QPushButton(self.page_6)
        self.puzzleGame2.setGeometry(QtCore.QRect(0, 214, 149, 23))
        self.puzzleGame2.setObjectName("puzzleGame2")
        self.runningTime = QtWidgets.QLabel(self.page_6)
        self.runningTime.setGeometry(QtCore.QRect(0, 5, 150, 15))
        self.runningTime.setObjectName("runningTime")
        self.searchDepth = QtWidgets.QLabel(self.page_6)
        self.searchDepth.setGeometry(QtCore.QRect(0, 25, 150, 15))
        self.searchDepth.setObjectName("searchDepth")
        self.expandedNodes = QtWidgets.QLabel(self.page_6)
        self.expandedNodes.setGeometry(QtCore.QRect(0, 45, 150, 15))
        self.expandedNodes.setObjectName("expandedNodes")
        self.frontierNodes = QtWidgets.QLabel(self.page_6)
        self.frontierNodes.setGeometry(QtCore.QRect(0, 65, 150, 15))
        self.frontierNodes.setObjectName("frontierNodes")
        self.curStep = QtWidgets.QSpinBox(self.page_6)
        self.curStep.setGeometry(QtCore.QRect(1, 90, 73, 23))
        self.curStep.setObjectName("curStep")
        self.curStep.setAlignment(QtCore.Qt.AlignCenter)
        self.jmpStep = QtWidgets.QPushButton(self.page_6)
        self.jmpStep.setGeometry(QtCore.QRect(75, 90, 74, 23))
        self.jmpStep.setObjectName("jmpStep")
        self.stackedWidget.addWidget(self.page_6)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(420, 80, 71, 15))
        self.label_2.setObjectName("label_2")
        exp1.setCentralWidget(self.centralwidget)

        self.retranslateUi(exp1)
        self.stackedWidget.setCurrentIndex(0)

        QtCore.QMetaObject.connectSlotsByName(exp1)

    def retranslateUi(self, exp1):
        _translate = QtCore.QCoreApplication.translate
        exp1.setWindowTitle(_translate("exp1", "2020052054-田翔熙-实验一"))
        self.img1.setPixmap(QtGui.QPixmap("img/1.png"))
        self.img2.setPixmap(QtGui.QPixmap("img/2.png"))
        self.img3.setPixmap(QtGui.QPixmap("img/0.png"))
        self.img4.setPixmap(QtGui.QPixmap("img/3.png"))
        self.img5.setPixmap(QtGui.QPixmap("img/4.png"))
        self.img6.setPixmap(QtGui.QPixmap("img/5.png"))
        self.img7.setPixmap(QtGui.QPixmap("img/6.png"))
        self.img8.setPixmap(QtGui.QPixmap("img/7.png"))
        self.img9.setPixmap(QtGui.QPixmap("img/8.png"))
        self.startState.setText(_translate("exp1", "123456789"))
        self.startState.setValidator(QtGui.QRegularExpressionValidator(QtCore.QRegularExpression("[0-9]+$")))
        self.goalState.setText(_translate("exp1", "124567893"))
        self.goalState.setValidator(QtGui.QRegularExpressionValidator(QtCore.QRegularExpression("[0-9]+$")))
        self.label.setText(_translate("exp1", "初始状态"))
        self.shuffleButton.setText(_translate("exp1", "随机生成目标状态"))
        self.shuffleButton.clicked.connect(shuffleGoal)
        self.label_15.setText(_translate("exp1", "算法选择"))
        self.byStep.setText(_translate("exp1", "按步查看"))
        self.byStep.clicked.connect(lambda : self.stackedWidget.setCurrentIndex(1))
        self.startSearch.setText(_translate("exp1", "开始搜索"))
        self.startSearch.clicked.connect(search)
        self.puzzleButton1.setText(_translate("exp1", "拼图游戏"))
        self.bfs.setText(_translate("exp1", "广度优先搜索（BFS）"))
        self.astar.setText(_translate("exp1", "A*搜索（AStar）"))
        self.back.setText(_translate("exp1", "返回"))
        self.back.clicked.connect(lambda : self.stackedWidget.setCurrentIndex(0))
        self.preStep.setText(_translate("exp1", "上一步"))
        self.preStep.clicked.connect(lambda : stepJump('-1'))
        self.nextStep.setText(_translate("exp1", "下一步"))
        self.nextStep.clicked.connect(lambda : stepJump('+1'))
        self.puzzleGame2.setText(_translate("exp1", "拼图游戏"))
        self.runningTime.setText(_translate("exp1", "总用时"))
        self.searchDepth.setText(_translate("exp1", "搜索深度"))
        self.expandedNodes.setText(_translate("exp1", "扩展结点"))
        self.frontierNodes.setText(_translate("exp1", "边界结点"))
        self.curStep.setSuffix(_translate("exp1", "步"))
        self.curStep.setPrefix(_translate("exp1", "第"))
        self.jmpStep.setText(_translate("exp1", "跳转"))
        self.jmpStep.clicked.connect(lambda : stepJump(self.curStep.text()))
        self.label_2.setText(_translate("exp1", "目标状态"))


class State:

    def __init__(self, state, parent, move_from, depth, cost):
        self.state = state
        self.parent = parent
        self.move_from = move_from
        self.depth = depth
        self.cost = cost
        if self.state:
            self.map = ''.join(str(e) for e in self.state)

    def __eq__(self, other):
        return self.map == other.map

    def __lt__(self, other):
        return self.cost < other.cost


global goal_node
moves = []
max_frontier_size = 0
max_search_depth = 0
nodes_expanded = 0
QPixmaps = []
QImages = []
startState = [1, 2, 0, 3, 4, 5, 6, 7, 8]
goalState = [1, 2, 3, 4, 5, 6, 7, 8, 0]
transcode = {3: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}
rev_transcode = {v: k for k, v in transcode.items()}


def shuffleGoal():
    global goalState
    goalState = list(range(9))
    shuffle(goalState)
    ui.goalState.setText(''.join(map(str, [rev_transcode[i] for i in goalState])))


def hasSolve(state):
    num_sum = 0
    for i in range(0, len(state)):
        if state[i] == 0:
            continue
        else:
            for j in range(0, i):
                if state[j] > state[i]:
                    num_sum += 1
    return num_sum % 2 == 0


def search():
    global startState, goalState, max_frontier_size, max_search_depth, nodes_expanded
    startState = [transcode[int(i)] for i in ui.startState.text()]
    goalState = [transcode[int(i)] for i in ui.goalState.text()]
    if not hasSolve(goalState):
        QtWidgets.QMessageBox.warning(ui.centralwidget, "错误", "目标状态无解！")
        return
    max_frontier_size = 0
    max_search_depth = 0
    nodes_expanded = 0
    moves.clear()
    ui.preStep.setEnabled(True)
    ui.stackedWidget.setEnabled(False)
    ui.goalState.setEnabled(False)
    start = timeit.default_timer()
    if ui.bfs.isChecked():
        frontier = bfs(startState, goalState)
    elif ui.astar.isChecked():
        frontier = astar(startState, goalState)
    stop = timeit.default_timer()
    ui.goalState.setEnabled(True)
    ui.stackedWidget.setEnabled(True)
    ui.stackedWidget.setCurrentIndex(1)
    ui.byStep.setEnabled(True)
    generateMoves()
    ui.runningTime.setText("总用时：" + format(stop-start,'.8f') + "s")
    ui.searchDepth.setText("搜索深度：" + str(goal_node.depth) + '/' + str(max_search_depth))
    ui.expandedNodes.setText("扩展结点：" + str(nodes_expanded))
    ui.frontierNodes.setText("边界结点：" + str(frontier) + '/' + str(max_frontier_size))
    ui.curStep.setMaximum(goal_node.depth)
    ui.curStep.setValue(goal_node.depth)
    ui.nextStep.setEnabled(False)


def generateMoves():
    global moves
    current_node = goal_node
    while startState != current_node.state:
        moves.insert(0, current_node.state)
        current_node = current_node.parent
    moves.insert(0, startState)


def stepJump(step):
    if step == '+1':
        if ui.curStep.value() < ui.curStep.maximum():
            ui.curStep.setValue(ui.curStep.value() + 1)
        if ui.curStep.value() == ui.curStep.maximum():
            ui.nextStep.setEnabled(False)
        ui.preStep.setEnabled(True)
    elif step == '-1':
        if ui.curStep.value() > ui.curStep.minimum():
            ui.curStep.setValue(ui.curStep.value() - 1)
        if ui.curStep.value() == ui.curStep.minimum():
            ui.preStep.setEnabled(False)
        ui.nextStep.setEnabled(True)
    else:
        if ui.curStep.value() == ui.curStep.maximum():
            ui.nextStep.setEnabled(False)
            ui.preStep.setEnabled(True)
        elif ui.curStep.value() == ui.curStep.minimum():
            ui.preStep.setEnabled(False)
            ui.nextStep.setEnabled(True)
        else:
            ui.preStep.setEnabled(True)
            ui.nextStep.setEnabled(True)
    for i in range(9):
        QImages[i].setPixmap(QPixmaps[moves[ui.curStep.value()][i]])


def bfs(start_state, goal_state):
    global max_frontier_size, goal_node, max_search_depth  # 最大扩展结点数，目标结点，最大搜索深度
    explored = set()  # 已经探索过的状态
    queue = deque([State(start_state, None, None, 0, 0)])  # 队列
    while queue:  # 队列不为空
        node = queue.popleft()  # 取出队列中的第一个状态
        explored.add(node.map)  # 将该状态加入已经探索过的状态中
        stateView(node.state)  # 显示状态
        if node.state == goal_state:  # 如果该状态为目标状态
            goal_node = node  # 将该状态赋值给goal_node
            frontier = len(queue)  # 边界结点数
            queue.clear()  # 清空队列
            return frontier  # 返回边界结点数
        neighbors = expand(node)  # 扩展该状态
        for neighbor in neighbors:  # 遍历扩展出的状态
            if neighbor.map not in explored:  # 如果该状态没有被探索过
                queue.append(neighbor)  # 将该状态加入队列
                explored.add(neighbor.map)  # 将该状态加入已经探索过的状态中
                if neighbor.depth > max_search_depth:  # 如果该状态的深度大于最大搜索深度
                    max_search_depth += 1  # 最大搜索深度加1
        if len(queue) > max_frontier_size:  # 如果队列的长度大于最大扩展结点数
            max_frontier_size = len(queue)  # 最大扩展结点数等于队列的长度


def astar(start_state, goal_state):
    global max_frontier_size, goal_node, max_search_depth  # 最大扩展结点数，目标结点，最大搜索深度
    explored = set()  # 已经探索过的状态
    heap = list()  # 优先队列
    heap_entry = dict()  # 优先队列中的状态
    cost = estimate_cost(start_state)  # 初次估计的代价
    root = State(start_state, None, None, 0, cost)  # 初始状态
    heappush(heap, root)  # 将初始状态加入优先队列
    heap_entry[root.map] = root  # 建立状态索引
    while heap:  # 当优先队列不为空时
        node = heappop(heap)  # 弹出优先队列中代价最小的状态
        explored.add(node.map)  # 将弹出的状态加入已探索的状态中
        stateView(node.state)  # 显示状态
        if node.state == goal_state:  # 如果弹出的状态为目标状态
            goal_node = node  # 将目标状态赋值给goal_node
            frontier = len(heap)  # 边界结点数
            heap.clear()  # 清空优先队列
            return frontier  # 返回边界结点数
        neighbors = expand(node)  # 扩展弹出的状态
        for neighbor in neighbors:  # 遍历扩展出的状态
            neighbor.cost = neighbor.depth + estimate_cost(neighbor.state)  # 计算每个状态的代价
            if neighbor.map not in explored:  # 如果该状态没有被探索过
                heappush(heap, neighbor)  # 将该状态加入优先队列
                explored.add(neighbor.map)  # 将该状态加入已探索的状态中
                heap_entry[neighbor.map] = neighbor  # 建立状态索引
                if neighbor.depth > max_search_depth:  # 如果该状态的深度大于达到过的最大深度
                    max_search_depth += 1  # 将达到过的最大深度加1
            elif neighbor.map in heap_entry and neighbor.cost < heap_entry[neighbor.map].cost:
                # 如果该状态已经被探索过，且该状态的代价小于上次探索该状态的代价
                heap_index = heap.index((heap_entry[neighbor.map]))  # 定位状态
                heap[int(heap_index)] = neighbor  # 更新状态
                heap_entry[neighbor.map] = neighbor  # 更新状态
                heapify(heap)  # 重新调整优先队列
        if len(heap) > max_frontier_size:  # 如果优先队列的长度大于达到过的最大边界结点数量
            max_frontier_size = len(heap)  # 将达到过的最大边界结点数量赋值为优先队列的长度


def expand(node):
    global nodes_expanded  # 已经扩展过的结点数量
    nodes_expanded += 1  # 已经扩展过的结点数量加1
    neighbors = list()  # 存储扩展出的状态
    neighbors.append(State(move(node.state, 1), node, 1, node.depth + 1, 0))  # 将上移后的状态加入neighbors
    neighbors.append(State(move(node.state, 2), node, 2, node.depth + 1, 0))  # 将下移后的状态加入neighbors
    neighbors.append(State(move(node.state, 3), node, 3, node.depth + 1, 0))  # 将左移后的状态加入neighbors
    neighbors.append(State(move(node.state, 4), node, 4, node.depth + 1, 0))  # 将右移后的状态加入neighbors
    nodes = [neighbor for neighbor in neighbors if neighbor.state]  # 将neighbors中的空状态去除
    return nodes  # 返回neighbors


def move(state, position):
    new_state = state[:]
    index = new_state.index(0)
    if position == 1:  # Up
        if index not in range(0, 3):
            temp = new_state[index - 3]
            new_state[index - 3] = new_state[index]
            new_state[index] = temp
            return new_state
        else:
            return None
    if position == 2:  # Down
        if index not in range(6, 9):
            temp = new_state[index + 3]
            new_state[index + 3] = new_state[index]
            new_state[index] = temp
            return new_state
        else:
            return None
    if position == 3:  # Left
        if index not in range(0, 9, 3):
            temp = new_state[index - 1]
            new_state[index - 1] = new_state[index]
            new_state[index] = temp
            return new_state
        else:
            return None
    if position == 4:  # Right
        if index not in range(3 - 1, 9, 3):
            temp = new_state[index + 1]
            new_state[index + 1] = new_state[index]
            new_state[index] = temp
            return new_state
        else:
            return None


def estimate_cost(state):
    cost = 0
    for i in range(1, 9):
        cost += (abs(state.index(i) % 3 - goalState.index(i) % 3)
                 + abs(state.index(i) // 3 - goalState.index(i) // 3))
    return cost


def stateView(state):
    for i in range(9):
        QImages[i].setPixmap(QPixmaps[state[i]])
    QtWidgets.QApplication.processEvents()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
    for i in range(9):
        QPixmaps.append(QtGui.QPixmap("img/" + str(i) + ".png"))
    ui = Ui_exp1()
    ui.setupUi(window)
    QImages.append(ui.img1)
    QImages.append(ui.img2)
    QImages.append(ui.img3)
    QImages.append(ui.img4)
    QImages.append(ui.img5)
    QImages.append(ui.img6)
    QImages.append(ui.img7)
    QImages.append(ui.img8)
    QImages.append(ui.img9)
    window.show()
    sys.exit(app.exec())
