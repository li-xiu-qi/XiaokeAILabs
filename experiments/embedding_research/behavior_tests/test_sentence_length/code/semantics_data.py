# -*- coding: utf-8 -*-
"""补做四的语义粒度语料（2026-09-26）。

三个层次，全部与主实验的 20 主题体系平行：

1. WORD_GROUPS  词级：同义组、反义组、上下位、共现对、无关对、跨语言对应词。
                 测「词与词」的相似度，检验经典假设：语境共现压过语义相反，
                 使反义词的 embedding 相似度高于无关词。
2. PARA_VARIANTS 句子表达谱：同一意思的四种说法（口语、书面、简化、详细），
                 中英平行。测「句子与句子」：模型对同一命题不同表述的鲁棒边界。
3. PARA_EXTRA    段落粒度的同主题短句：每主题 c / d 两句，与 themes 的 a、a_rel
                 合成 1 到 4 段的同主题文本，测「文章与文章」的长度饱和曲线。

构造约束：
  - 变体只改表达，不引入新命题；brief 允许丢失修饰，verbose 允许补充细节
  - c / d 是同一主题的独立短句，不与 a、a_rel 重复用词超过必要限度
  - 全部中英平行，便于跨语言对照
"""

# ---------------------------------------------------------------- 词级

# 语义组。同组内两两视为同义；相邻组标注关系。
# rel 取值：syn 同义 / ant 反义 / hyper 上下位 / cooc 共现不对等 / unrelated 无关 / cross 跨语言同义
WORD_GROUPS_ZH = [
    {"gid": "joy", "rel": "syn", "words": ["快乐", "高兴", "喜悦", "愉快"]},
    {"gid": "sorrow", "rel": "ant", "words": ["悲伤", "难过", "痛苦", "沮丧"]},
    {"gid": "big", "rel": "syn", "words": ["巨大", "庞大", "硕大", "宏大"]},
    {"gid": "small", "rel": "ant", "words": ["微小", "细小", "渺小", "微弱"]},
    {"gid": "buy", "rel": "syn", "words": ["购买", "买入", "采购", "添置"]},
    {"gid": "sell", "rel": "ant", "words": ["出售", "卖出", "售卖", "转让"]},
    {"gid": "fast", "rel": "syn", "words": ["迅速", "快速", "迅捷", "飞快"]},
    {"gid": "slow", "rel": "ant", "words": ["缓慢", "迟缓", "拖沓"]},
    {"gid": "dog", "rel": "syn", "words": ["小狗", "犬只", "犬类", "狗"]},
    {"gid": "animal", "rel": "hyper", "words": ["动物", "生物"]},
]

WORD_GROUPS_EN = [
    {"gid": "joy", "rel": "syn", "words": ["happy", "glad", "joyful", "cheerful"]},
    {"gid": "sorrow", "rel": "ant", "words": ["sad", "unhappy", "sorrowful", "depressed"]},
    {"gid": "big", "rel": "syn", "words": ["huge", "enormous", "massive", "gigantic"]},
    {"gid": "small", "rel": "ant", "words": ["tiny", "small", "minute", "miniature"]},
    {"gid": "buy", "rel": "syn", "words": ["buy", "purchase", "acquire"]},
    {"gid": "sell", "rel": "ant", "words": ["sell", "vend", "trade"]},
    {"gid": "fast", "rel": "syn", "words": ["fast", "rapid", "quick", "swift"]},
    {"gid": "slow", "rel": "ant", "words": ["slow", "sluggish", "unhurried"]},
    {"gid": "dog", "rel": "syn", "words": ["dog", "canine", "hound", "puppy"]},
    {"gid": "animal", "rel": "hyper", "words": ["animal", "creature", "organism"]},
]

# 共现强但语义不对等的词对：分布语境高度重合，语义不同类
COOC_PAIRS_ZH = [("医生", "医院"), ("学生", "学校"), ("雨", "伞"), ("咖啡", "杯子")]
COOC_PAIRS_EN = [("doctor", "hospital"), ("student", "school"), ("rain", "umbrella"), ("coffee", "cup")]

# 无关词对：跨组、跨语义域
UNRELATED_PAIRS_ZH = [("快乐", "地铁"), ("巨大", "采购"), ("悲伤", "迅捷"), ("狗", "出售")]
UNRELATED_PAIRS_EN = [("happy", "subway"), ("huge", "purchase"), ("sad", "swift"), ("dog", "sell")]

# 跨语言同义对：同一概念的两种语言
CROSS_PAIRS = [
    ("狗", "dog"), ("快乐", "happy"), ("巨大", "huge"),
    ("购买", "buy"), ("出售", "sell"), ("医生", "doctor"),
    ("学校", "school"), ("迅速", "rapid"),
]

# 反义对照的两组配对（同语义域、意义相反）
ANT_GROUP_PAIRS = [("joy", "sorrow"), ("big", "small"), ("buy", "sell"), ("fast", "slow")]


# ------------------------------------------------- 句子表达谱（中英平行）

# 每主题五个版本：orig 取自 themes 的 a 句；oral 口语化；formal 书面化；
# brief 删修饰；verbose 同命题加长。只改表达，不引入新命题。
PARA_VARIANTS = [
    {
        "id": "pet",
        "oral": "那小狗正在草地上追球玩儿呢。",
        "formal": "一只犬科动物正在草地上追逐球体。",
        "brief": "狗追球。",
        "verbose": "午后，邻居家那只三个月大的小狗正在门前那片修剪整齐的草地上，全神贯注地追着一个脏兮兮的网球。",
        "en_oral": "That dog's chasing a ball around on the grass.",
        "en_formal": "A canine is in pursuit of a ball upon the lawn.",
        "en_brief": "Dog chases ball.",
        "en_verbose": "In the afternoon, the neighbors' three-month-old puppy is focused on chasing a muddy tennis ball across the neatly trimmed lawn in front of the house.",
    },
    {
        "id": "programming",
        "oral": "学编程真的挺有用的。",
        "formal": "掌握编程技能具有显著的实用价值。",
        "brief": "编程有用。",
        "verbose": "系统学习编程不仅能带来直接的就业机会，更重要的是它强迫人把模糊的需求拆解成计算机可以执行的精确步骤，这种训练会迁移到日常问题的处理上。",
        "en_oral": "Learning to code is actually pretty useful.",
        "en_formal": "Proficiency in programming carries substantial practical value.",
        "en_brief": "Coding is useful.",
        "en_verbose": "Studying programming systematically not only opens direct career opportunities but, more importantly, forces one to decompose vague requirements into precise steps a computer can execute, and that training transfers to everyday problem solving.",
    },
    {
        "id": "exam",
        "oral": "他正在复习备考呢。",
        "formal": "该生正在为即将到来的考试进行系统性复习。",
        "brief": "他备考。",
        "verbose": "距离考试还有两周，他把过去三年的真题按知识点分类整理，每天上午专攻薄弱章节，晚上用错题本复盘当天做错的题目。",
        "en_oral": "He's getting ready for the exam.",
        "en_formal": "The student is engaged in systematic revision for an upcoming examination.",
        "en_brief": "He is studying for a test.",
        "en_verbose": "Two weeks before the exam, he sorted three years of past papers by topic, attacking weak chapters every morning and reviewing the day's mistakes from an error notebook each evening.",
    },
    {
        "id": "health",
        "oral": "多动动对身体有好处。",
        "formal": "体育锻炼对健康状况具有积极影响。",
        "brief": "运动利健康。",
        "verbose": "大量追踪研究一致显示，每周累计一百五十分钟中等强度运动的人群，心血管事件风险和全因死亡率都显著低于久坐人群，而且大部分收益不需要高强度训练也能获得。",
        "en_oral": "Moving around more is good for you.",
        "en_formal": "Physical exercise exerts a beneficial effect on health status.",
        "en_brief": "Exercise helps health.",
        "en_verbose": "A large body of cohort research consistently shows that people accumulating one hundred fifty minutes of moderate activity per week have markedly lower cardiovascular event rates and all-cause mortality than sedentary people, and most of the benefit arrives without high-intensity training.",
    },
    {
        "id": "weather",
        "oral": "外头下大雨了。",
        "formal": "当日出现降水天气。",
        "brief": "下雨了。",
        "verbose": "从清晨起云层就压得很低，到中午终于落下密实的雨点，落在空调外机上连成一片白噪音，阳台的地砖很快湿成深色。",
        "en_oral": "It's pouring outside.",
        "en_formal": "Precipitation occurred on the given day.",
        "en_brief": "It is raining.",
        "en_verbose": "Since early morning the clouds hung low, and by noon dense raindrops finally fell, drumming on the outdoor air conditioner into a wash of white noise, the balcony tiles darkening quickly.",
    },
    {
        "id": "transit",
        "oral": "坐地铁挺省事的。",
        "formal": "轨道交通系统具备较高的通行效率与可靠性。",
        "brief": "地铁方便。",
        "verbose": "地铁在地下专用通道里运行，不受红绿灯和堵车影响，早高峰从城西到城东四十分钟内可达，而同距离打车常常要一个小时以上。",
        "en_oral": "Taking the subway saves a lot of trouble.",
        "en_formal": "The rail transit system demonstrates high transit efficiency and reliability.",
        "en_brief": "The subway is convenient.",
        "en_verbose": "The subway runs in dedicated underground corridors, untouched by traffic lights and jams; forty minutes gets you from the west to the east side in the morning rush, while the same distance by taxi often takes over an hour.",
    },
    {
        "id": "gadget",
        "oral": "这手机电池挺抗用的。",
        "formal": "该机型在续航表现上具备优势。",
        "brief": "手机续航长。",
        "verbose": "五千毫安时的电池配合功耗控制，中度使用能撑到第二天晚上，而且半小时快充能回六成电量，出差时基本不用带充电宝。",
        "en_oral": "This phone's battery really lasts.",
        "en_formal": "This model holds an advantage in battery endurance.",
        "en_brief": "Long battery life.",
        "en_verbose": "A five-thousand-milliamp-hour battery with careful power management carries through moderate use into the second evening, and a half-hour fast charge recovers sixty percent, so business trips rarely need a power bank.",
    },
    {
        "id": "books",
        "oral": "这本小说看起来挺上瘾的。",
        "formal": "该作品叙事结构严密，情节推进紧凑。",
        "brief": "情节紧凑。",
        "verbose": "作者每三章设一次反转，明线是追查真相，暗线是主角与父亲的和解，两条线在最后一章合流，读完后半部分时几乎停不下来。",
        "en_oral": "This novel is a real page-turner.",
        "en_formal": "The work features a rigorous narrative structure with tightly advanced plotting.",
        "en_brief": "Tight plot.",
        "en_verbose": "The author sets a reversal every three chapters, the visible line a hunt for truth and the hidden one the protagonist's reconciliation with his father; the two merge in the final chapter, and the second half is almost impossible to put down.",
    },
    {
        "id": "work",
        "oral": "项目现在推进得挺顺的。",
        "formal": "项目当前进度符合既定计划。",
        "brief": "进度正常。",
        "verbose": "两个关键里程碑已按期交付，剩余测试任务排入下周计划，风险集中在第三方接口联调上，已经安排了备选方案，整体可控。",
        "en_oral": "The project's moving along pretty smoothly.",
        "en_formal": "The project's current progress conforms to the established plan.",
        "en_brief": "On schedule.",
        "en_verbose": "Two key milestones were delivered on time, remaining tests are scheduled into next week, and risk concentrates on third-party integration, where a fallback has been arranged; overall it stays under control.",
    },
    {
        "id": "travel",
        "oral": "湖边风景挺好的。",
        "formal": "该湖区自然景观具有良好的观赏价值。",
        "brief": "湖边风景好。",
        "verbose": "傍晚的湖面被落日染成暖色，柳条垂到水面，散步的人放慢脚步，远处渔船的轮廓慢慢变成剪影。",
        "en_oral": "The lake looks really nice.",
        "en_formal": "The lake district's natural landscape holds considerable aesthetic value.",
        "en_brief": "Nice lake view.",
        "en_verbose": "At dusk the lake surface takes the warm color of the setting sun, willows touch the water, strollers slow their pace, and the silhouette of a fishing boat gradually settles into place.",
    },
]

# ------------------------------------------------- 段落粒度的同主题短句

# 每主题 c / d 两句独立短句，与 a、a_rel 不共用整句，用于拼 1 到 4 段的同主题文本
PARA_EXTRA = [
    {
        "id": "pet",
        "c": "它最喜欢把玩具叼回来。",
        "d": "每次出门它都兴奋得转圈。",
        "en_c": "It loves carrying its toys back.",
        "en_d": "It spins with excitement every time we head out.",
    },
    {
        "id": "programming",
        "c": "它训练逻辑思维。",
        "d": "入门可以从脚本开始。",
        "en_c": "It trains logical thinking.",
        "en_d": "Beginners can start with small scripts.",
    },
    {
        "id": "exam",
        "c": "书桌上堆满了资料。",
        "d": "他定了复习计划。",
        "en_c": "His desk is piled with materials.",
        "en_d": "He made a study plan.",
    },
    {
        "id": "health",
        "c": "走路也算一种运动。",
        "d": "关键是长期坚持。",
        "en_c": "Walking counts as exercise.",
        "en_d": "Consistency matters most.",
    },
    {
        "id": "weather",
        "c": "天空灰蒙蒙的。",
        "d": "气温降了几度。",
        "en_c": "The sky is grey.",
        "en_d": "The temperature dropped a few degrees.",
    },
    {
        "id": "transit",
        "c": "它不受路面影响。",
        "d": "换乘线路覆盖广。",
        "en_c": "It is unaffected by road traffic.",
        "en_d": "Its transfer network covers a wide area.",
    },
    {
        "id": "gadget",
        "c": "充电速度也快。",
        "d": "重量控制得不错。",
        "en_c": "It also charges fast.",
        "en_d": "The weight is well managed.",
    },
    {
        "id": "books",
        "c": "人物动机清楚。",
        "d": "结局没有烂尾。",
        "en_c": "The characters' motives are clear.",
        "en_d": "The ending holds up.",
    },
    {
        "id": "work",
        "c": "没有阻塞项。",
        "d": "下周可以提测。",
        "en_c": "There are no blockers.",
        "en_d": "Testing can start next week.",
    },
    {
        "id": "travel",
        "c": "傍晚最适合散步。",
        "d": "岸边有柳树。",
        "en_c": "Evening suits a stroll best.",
        "en_d": "Willows line the shore.",
    },
]
