# -*- coding: utf-8 -*-
"""20 个主题的中英平行句对数据。

每个主题包含：
  a, b          基础短句对（同主题、不同侧重），长度尽量接近
  a_rel, b_rel  主题相关的加长句（拼接在 a / b 后）
  en_*          上面四个字段的英文平行版本

无关加长素材统一从 NOISE_POOL 取（中英平行），与全部 20 个主题都不相关。

设计约束：
  - a/b 长短接近，避免长度差本身成为混杂
  - rel 加长只补充主题内细节，不引入新主题词
  - irr 加长引入完全不同域的句子
"""

THEMES = [
    {
        "id": "pet",
        "a": "小狗在草地上追球。",
        "b": "猫在窗台上睡觉。",
        "a_rel": "小狗在公园的草地上兴奋地追着一个红色的皮球，跑几步就停下来喘气。",
        "b_rel": "猫蜷在午后的窗台上晒着太阳，尾巴偶尔轻轻摆动两下。",
        "en_a": "The dog is chasing a ball on the grass.",
        "en_b": "The cat is sleeping on the windowsill.",
        "en_a_rel": "The dog excitedly chases a red ball across the park lawn, stopping every few steps to catch its breath.",
        "en_b_rel": "The cat curls up on the afternoon windowsill in the sun, its tail flicking gently now and then.",
    },
    {
        "id": "programming",
        "a": "学习编程很有用。",
        "b": "它能解决计算问题。",
        "a_rel": "学习编程能训练把大问题拆成小步骤的思维方式，对日常决策也有帮助。",
        "b_rel": "编写脚本可以把重复的计算和格式整理工作交给电脑自动完成。",
        "en_a": "Learning programming is very useful.",
        "en_b": "It can solve computational problems.",
        "en_a_rel": "Learning to code trains the habit of breaking big problems into small steps, which helps with everyday decisions too.",
        "en_b_rel": "Writing scripts hands repetitive computation and formatting work over to the computer to finish automatically.",
    },
    {
        "id": "exam",
        "a": "他在准备考试。",
        "b": "她在复习笔记。",
        "a_rel": "他在图书馆里整理错题，把容易混淆的知识点做成卡片反复过。",
        "b_rel": "她在房间里按章节梳理课堂笔记，用不同颜色的笔标出重点和疑问。",
        "en_a": "He is preparing for an exam.",
        "en_b": "She is reviewing her notes.",
        "en_a_rel": "In the library he is sorting through wrong answers, turning easily confused points into flashcards to review again and again.",
        "en_b_rel": "In her room she is going through the lecture notes chapter by chapter, marking key points and open questions in different colors.",
    },
    {
        "id": "health",
        "a": "运动有益健康。",
        "b": "饮食均衡也很重要。",
        "a_rel": "每周坚持几次适度运动，能增强心肺功能并改善睡眠质量。",
        "b_rel": "每餐保证蔬菜和蛋白质的搭配，能为身体提供稳定的营养供给。",
        "en_a": "Exercise is good for health.",
        "en_b": "A balanced diet matters just as much.",
        "en_a_rel": "Sticking to moderate exercise a few times a week improves cardiorespiratory fitness and sleep quality.",
        "en_b_rel": "Pairing vegetables with protein at every meal gives the body a steady supply of nutrients.",
    },
    {
        "id": "weather",
        "a": "今天下雨了。",
        "b": "出门记得带伞。",
        "a_rel": "从早上开始雨就没停，路面已经积起大片的水洼。",
        "b_rel": "带上一把结实的伞，风大的时候伞面才不容易被吹翻。",
        "en_a": "It is raining today.",
        "en_b": "Remember to take an umbrella.",
        "en_a_rel": "The rain has not stopped since morning and large puddles have collected on the road.",
        "en_b_rel": "Take a sturdy umbrella, one whose canopy will not flip inside out in strong wind.",
    },
    {
        "id": "transit",
        "a": "地铁很方便。",
        "b": "公交常堵车。",
        "a_rel": "地铁准点率高，换乘通道指示清楚，早晚高峰也很少误点。",
        "b_rel": "公交高峰期常被堵在路口，一趟车有时要比平时多花二十分钟。",
        "en_a": "The subway is very convenient.",
        "en_b": "Buses often get stuck in traffic.",
        "en_a_rel": "The subway is punctual with clear transfer signs and rarely runs late even in the morning rush.",
        "en_b_rel": "Buses jam at intersections during rush hour and one trip can take twenty minutes longer than usual.",
    },
    {
        "id": "gadget",
        "a": "这款手机续航长。",
        "b": "新笔记本散热好。",
        "a_rel": "这款手机满电能用一整天，出差时不用随身带充电宝。",
        "b_rel": "新笔记本双风扇压在键盘上方，长时间编译时机身也只是温热。",
        "en_a": "This phone has long battery life.",
        "en_b": "The new laptop cools well.",
        "en_a_rel": "This phone lasts a full day on one charge, so a power bank is unnecessary on business trips.",
        "en_b_rel": "The new laptop runs dual fans near the keyboard, and the chassis stays barely warm during long compiles.",
    },
    {
        "id": "books",
        "a": "这本书情节紧凑。",
        "b": "那本文笔细腻。",
        "a_rel": "这本书每章结尾都留悬念，一口气读完三章还停不下来。",
        "b_rel": "那本散文集用词克制，写日常食事也能读出层次。",
        "en_a": "This book has a tight plot.",
        "en_b": "That one has delicate prose.",
        "en_a_rel": "Every chapter of this novel ends on a cliffhanger; three chapters passed before I could put it down.",
        "en_b_rel": "That essay collection uses restrained wording and finds layers inside ordinary descriptions of food.",
    },
    {
        "id": "work",
        "a": "项目进度正常。",
        "b": "团队沟通顺畅。",
        "a_rel": "两个关键里程碑都已交付，剩下的测试任务排进了下周计划。",
        "b_rel": "站会控制在十五分钟内，问题当面对齐，文档当天同步。",
        "en_a": "The project is on schedule.",
        "en_b": "Team communication is smooth.",
        "en_a_rel": "Two key milestones have been delivered, and the remaining tests are scheduled into next week.",
        "en_b_rel": "Stand-ups finish within fifteen minutes, issues get aligned face to face, and docs are synced the same day.",
    },
    {
        "id": "music_art",
        "a": "这首曲子旋律优美。",
        "b": "那幅画色彩浓烈。",
        "a_rel": "这首曲子的副歌在第三遍出现时转调，情绪的推力很明显。",
        "b_rel": "那幅画用大面积的朱红铺底，人物的轮廓几乎要从画布里跳出来。",
        "en_a": "This melody is beautiful.",
        "en_b": "That painting uses intense colors.",
        "en_a_rel": "The chorus modulates on its third appearance, and the emotional push is unmistakable.",
        "en_b_rel": "That painting lays out broad strokes of vermilion, and the figures seem ready to leap off the canvas.",
    },
    {
        "id": "travel",
        "a": "湖边风景宜人。",
        "b": "山顶空气清新。",
        "a_rel": "傍晚的湖边有风，柳条垂到水面，散步的人放慢了脚步。",
        "b_rel": "爬到山顶时云层在脚下，深吸一口气带着松针的味道。",
        "en_a": "The lakeside scenery is pleasant.",
        "en_b": "The mountain air is fresh.",
        "en_a_rel": "By evening the lake catches a breeze, willows touch the water, and strollers slow their pace.",
        "en_b_rel": "At the summit the clouds sit below foot, and one deep breath carries the smell of pine.",
    },
    {
        "id": "cooking",
        "a": "这道菜味道鲜美。",
        "b": "那个汤炖得香。",
        "a_rel": "这道菜出锅前淋一勺热油，蒜香被激起来，配米饭很合适。",
        "b_rel": "那个汤用小火煨了两个钟头，汤色发白，鲜味全进了汤里。",
        "en_a": "This dish tastes delicious.",
        "en_b": "That soup smells wonderful.",
        "en_a_rel": "A spoonful of hot oil poured over the dish before serving wakes up the garlic aroma and pairs well with rice.",
        "en_b_rel": "That soup simmered over low heat for two hours, turning milky, with all the savor locked into the broth.",
    },
    {
        "id": "fitness",
        "a": "跑步能增强体质。",
        "b": "游泳锻炼心肺。",
        "a_rel": "每周三次、每次半小时的慢跑，两个月下来静息心率会下降。",
        "b_rel": "游泳时呼吸节奏受划水限制，对心肺的负荷均匀而持续。",
        "en_a": "Running improves fitness.",
        "en_b": "Swimming trains the heart and lungs.",
        "en_a_rel": "Jogging half an hour three times a week lowers the resting heart rate within two months.",
        "en_b_rel": "In swimming the breath rhythm is tied to the stroke, giving the cardiorespiratory system an even, sustained load.",
    },
    {
        "id": "plants",
        "a": "玫瑰开得茂盛。",
        "b": "多肉长得喜人。",
        "a_rel": "这株玫瑰今年打了十几个花苞，浇水跟上后开得一层压一层。",
        "b_rel": "窗台上的多肉晒足了太阳，叶片饱满边上一层淡淡的红。",
        "en_a": "The roses are blooming lushly.",
        "en_b": "The succulents are growing well.",
        "en_a_rel": "This rose bush set more than a dozen buds this year, and with steady watering they open layer upon layer.",
        "en_b_rel": "The succulents on the windowsill get plenty of sun, their leaves plump with a faint red edge.",
    },
    {
        "id": "debug",
        "a": "这个 bug 很难复现。",
        "b": "那段代码逻辑清晰。",
        "a_rel": "同样的输入偶发触发空指针，加了日志跑一百次也只出现两回。",
        "b_rel": "那段代码把边界判断收在一个函数里，读起来不用来回翻上下文。",
        "en_a": "This bug is hard to reproduce.",
        "en_b": "That code is logically clear.",
        "en_a_rel": "The same input triggers a null pointer only occasionally; a hundred logged runs hit it twice.",
        "en_b_rel": "That code gathers the boundary checks into a single function, so reading it needs no jumping back and forth.",
    },
    {
        "id": "classroom",
        "a": "老师讲得很清楚。",
        "b": "学生提问很积极。",
        "a_rel": "老师用三个例子把抽象的定理串起来，板书从头到尾保持着结构。",
        "b_rel": "下课时学生围着讲台追问，把没懂的地方当场问清楚才离开。",
        "en_a": "The teacher explains clearly.",
        "en_b": "The students ask questions actively.",
        "en_a_rel": "The teacher threads an abstract theorem through three examples, keeping the board structured from start to end.",
        "en_b_rel": "Students crowd the podium after class, pressing their doubts until every part is cleared before leaving.",
    },
    {
        "id": "checkup",
        "a": "定期体检有必要。",
        "b": "规律作息很重要。",
        "a_rel": "很多早期异常没有症状，一年一次的全面检查能提前发现。",
        "b_rel": "固定就寝时间让激素节律稳定，第二天的精神状态差别明显。",
        "en_a": "Regular checkups are necessary.",
        "en_b": "A regular schedule matters a lot.",
        "en_a_rel": "Many early abnormalities show no symptoms, and an annual comprehensive exam catches them ahead of time.",
        "en_b_rel": "A fixed bedtime keeps hormone rhythms stable, and the difference in next-day alertness is obvious.",
    },
    {
        "id": "eco",
        "a": "随手关灯能省电。",
        "b": "少用塑料袋更环保。",
        "a_rel": "离开房间时关灯关空调，一年累积下来的电费相当可观。",
        "b_rel": "自带布袋去超市，一个家庭一年能少用几百个塑料袋。",
        "en_a": "Switching off lights saves power.",
        "en_b": "Using fewer plastic bags is greener.",
        "en_a_rel": "Turning off lights and air conditioning when leaving a room adds up to a considerable saving over a year.",
        "en_b_rel": "Bringing your own bags to the supermarket saves a household hundreds of plastic bags every year.",
    },
    {
        "id": "cinema",
        "a": "这部电影节奏紧凑。",
        "b": "那部结局意外。",
        "a_rel": "这部电影两小时里没有一场多余的戏，每段对话都在推进信息。",
        "b_rel": "那部片最后十分钟反转，回头看第二遍处处都是铺垫。",
        "en_a": "This movie has a tight pace.",
        "en_b": "That one ends unexpectedly.",
        "en_a_rel": "Over two hours this film carries no wasted scene, every exchange pushing the story forward.",
        "en_b_rel": "That film turns in the final ten minutes, and the second viewing reveals the hints everywhere.",
    },
    {
        "id": "gaming",
        "a": "这款游戏操作流畅。",
        "b": "那个关卡设计巧妙。",
        "a_rel": "这款游戏输入延迟低，连招判定跟手，搓招几乎没有挫败感。",
        "b_rel": "那个关卡把新机制藏在三条可选路线里，玩家自己发现时才恍然大悟。",
        "en_a": "This game controls smoothly.",
        "en_b": "That level is cleverly designed.",
        "en_a_rel": "This game has low input latency and responsive combo timing, so executing moves rarely feels frustrating.",
        "en_b_rel": "That level hides its new mechanic inside three optional routes, letting players discover it themselves.",
    },
]

# 无关加长素材双池：A 池给第一侧、B 池给第二侧，两侧噪声句不同。
# 否则双侧噪声条件会退化成「两端拼接同一段文本」，测到的是内容共享效应。
# 两条池都与全部 20 个主题无关。
NOISE_POOL_A = [
    {"zh": "火山口持续冒出浓烟。", "en": "The volcano keeps billowing thick smoke."},
    {"zh": "超市周末全场八折。", "en": "The supermarket has a twenty-percent storewide discount this weekend."},
    {"zh": "邻居在重新装修客厅。", "en": "The neighbors are renovating their living room."},
    {"zh": "快递已经放到驿站。", "en": "The parcel has been dropped off at the pickup station."},
    {"zh": "股市今天大幅低开。", "en": "The stock market opened sharply lower today."},
    {"zh": "候鸟开始往南迁徙。", "en": "Migratory birds have begun their journey south."},
    {"zh": "剧院今晚进行彩排。", "en": "The theater is holding a rehearsal tonight."},
    {"zh": "电台下周改版节目单。", "en": "The radio station is revising its schedule next week."},
    {"zh": "渔民凌晨出海捕鱼。", "en": "Fishermen set out to sea before dawn."},
    {"zh": "天文台预报有流星雨。", "en": "The observatory forecasts a meteor shower."},
    {"zh": "考古队挖出陶片。", "en": "The archaeology team unearthed pottery shards."},
    {"zh": "地铁三号线临时封站。", "en": "Line 3 of the subway is temporarily closed."},
    {"zh": "音乐节在调试音响。", "en": "The music festival is sound-checking."},
    {"zh": "村里正在办庙会。", "en": "The village is holding its temple fair."},
    {"zh": "海事台发布大风预警。", "en": "The maritime bureau issued a gale warning."},
]

NOISE_POOL_B = [
    {"zh": "新款无人机开始预售。", "en": "The new drone model has opened preorders."},
    {"zh": "剧团下月巡演城市公布。", "en": "The troupe announced its tour cities for next month."},
    {"zh": "图书馆延长了开放时间。", "en": "The library has extended its opening hours."},
    {"zh": "早市上西瓜卖得正好。", "en": "Watermelons are selling well at the morning market."},
    {"zh": "台风路径图又更新了。", "en": "The typhoon track map has been updated again."},
    {"zh": "小区加装了充电桩。", "en": "The neighborhood has installed charging stations."},
    {"zh": "今年羽绒服提前上架。", "en": "This year's down jackets hit the shelves early."},
    {"zh": "水管工约在下午上门。", "en": "The plumber is scheduled to come this afternoon."},
    {"zh": "港口积压了不少集装箱。", "en": "A backlog of containers has piled up at the port."},
    {"zh": "新版地图加入了街景。", "en": "The new map release added street-level imagery."},
    {"zh": "快递柜换了新的取件码。", "en": "The parcel locker issued a new pickup code."},
    {"zh": "本地姜价连续三周回落。", "en": "Local ginger prices have fallen for three straight weeks."},
    {"zh": "博物馆本周免预约参观。", "en": "The museum drops reservations this week."},
    {"zh": "夜班公交加密了班次。", "en": "Night buses have added extra departures."},
    {"zh": "卫星图显示冰盖缩小。", "en": "Satellite imagery shows the ice sheet shrinking."},
]


def noise_a_zh(idx: int) -> str:
    return NOISE_POOL_A[idx % len(NOISE_POOL_A)]["zh"]


def noise_a_en(idx: int) -> str:
    return NOISE_POOL_A[idx % len(NOISE_POOL_A)]["en"]


def noise_b_zh(idx: int) -> str:
    return NOISE_POOL_B[idx % len(NOISE_POOL_B)]["zh"]


def noise_b_en(idx: int) -> str:
    return NOISE_POOL_B[idx % len(NOISE_POOL_B)]["en"]
