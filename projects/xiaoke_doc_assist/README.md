### DeepSeek 智能文档助手应用说明文档

✨【你好，我是筱可，欢迎来到“筱可AI研习社”】✨

本次项目主题是：

**基于 Streamlit 和 DeepSeek 的智能文档助手开发实战**

🚀 标签关键词： AI实战派开发者 | 技术成长陪伴者 | RAG前沿探索者

🌈 期待与你：
成为"AI+成长"的双向奔赴伙伴！在这里你将获得：

- ✅ 每周更新的AI开发笔记
- ✅ 可复用的代码案例库
- ✅ 前沿技术应用场景拆解
- ✅ 学生党专属成长心法
- ✅ 定期精读好书

#### 应用概述

本应用是基于DeepSeek大模型的智能对话助手，支持以下核心功能：

- ✅ 多模型选择（提供3个不同规模的模型）
- ✅ 文档分析模式（支持TXT/PDF文件解析）
- ✅ 可调节的对话参数（温度值、上下文长度）
- ✅ 实时流式响应
- ✅ 对话历史管理

#### 功能特点

##### 1. 多模型支持

| 模型名称                          | 适用场景       |
| ----------------------------- | ---------- |
| DeepSeek-R1-Distill-Qwen-1.5B | 免费快速响应基础问答 |
| DeepSeek-V3                   | 通用场景对话     |
| DeepSeek-R1        | 复杂推理和长文本理解 |

##### 2. 文档分析模式

- 支持格式：TXT/PDF
- 处理流程：
  1. 上传文件自动解析
  2. 提取前500字符作为系统提示
  3. 保留最近18k字符上下文（可配置）
  4. 自动切换文档/普通模式

##### 3. 参数配置

- **Temperature** (0.001-1.2)：控制生成随机性
- **上下文长度** (100-30k tokens)：管理对话历史长度
- **最大输出长度** (固定8192 tokens)

#### 使用指南

##### 环境要求

```bash
Python 3.10+
依赖库：streamlit, pymupdf, python-dotenv, openai
```

##### 安装依赖

```
pip install streamlit PyMuPDF python-dotenv openai
```

##### 运行步骤

4. 配置环境变量
创建`.env`文件并写入api_key
注意：API_KEY来自硅基流动。

```env
API_KEY=your_deepseek_api_key
```

5. 启动应用

```bash
streamlit run xiaoke_doc_assist.py
```

##### 界面操作

6. 模型选择区（左侧）
   - 下拉选择模型版本
   - 滑动调节温度值
   - 设置上下文长度

7. 文档上传区
   - 支持拖拽上传
   - 自动识别文件类型
   - 成功加载后显示绿色提示

8. 对话界面
   - 用户输入框位于底部
   - 实时显示对话历史
   - 助理响应带打字机效果

#### 代码结构说明

##### 初始化流程图 (init_session 功能)

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="358pt" height="764pt" viewBox="0.00 0.00 357.68 764.20">
<g id="graph0" class="graph" transform="translate(4,760.2000122070312) scale(1)" data-name="初始化流程">

<polygon fill="white" stroke="none" points="-4,4 -4,-760.2 353.68,-760.2 353.68,4 -4,4" style=""/>
<!-- init_session -->
<g id="node1" class="node" pointer-events="visible" data-name="init_session">

<polygon fill="none" stroke="black" points="265.04,-756.2 173.15,-756.2 173.15,-720.2 265.04,-720.2 265.04,-756.2" style=""/>
<text text-anchor="middle" x="219.1" y="-734" font-family="SimSun" font-size="14.00" style="">init_session()</text>
</g>
<!-- check_messages -->
<g id="node2" class="node" pointer-events="visible" data-name="check_messages">

<polygon fill="none" stroke="black" points="324.79,-683.2 113.4,-683.2 113.4,-647.2 324.79,-647.2 324.79,-683.2" style=""/>
<text text-anchor="middle" x="219.1" y="-661" font-family="SimSun" font-size="14.00" style="">session_state 中是否存在 messages ?</text>
</g>
<!-- init_session&#45;&gt;check_messages -->
<g id="edge1" class="edge" data-name="init_session-&gt;check_messages">

<path fill="none" stroke="black" d="M219.1,-720.01C219.1,-712.43 219.1,-703.3 219.1,-694.74" style=""/>
<polygon fill="black" stroke="black" points="222.6,-694.74 219.1,-684.74 215.6,-694.74 222.6,-694.74" style=""/>
</g>
<!-- create_messages -->
<g id="node3" class="node" pointer-events="visible" data-name="create_messages">

<polygon fill="none" stroke="black" points="246.13,-594.4 64.06,-594.4 64.06,-558.4 246.13,-558.4 246.13,-594.4" style=""/>
<text text-anchor="middle" x="155.1" y="-572.2" font-family="SimSun" font-size="14.00" style="">st.session_state.messages = []</text>
</g>
<!-- check_messages&#45;&gt;create_messages -->
<g id="edge2" class="edge" data-name="check_messages-&gt;create_messages">

<path fill="none" stroke="black" d="M206.45,-647.05C197.39,-634.77 185.02,-617.99 174.69,-603.98" style=""/>
<polygon fill="black" stroke="black" points="177.58,-602 168.83,-596.03 171.95,-606.15 177.58,-602" style=""/>
<text text-anchor="middle" x="198.35" y="-616.6" font-family="Times,serif" font-size="14.00" style="">否</text>
</g>
<!-- check_uploaded_content -->
<g id="node4" class="node" pointer-events="visible" data-name="check_uploaded_content">

<polygon fill="none" stroke="black" points="346.94,-521.4 89.25,-521.4 89.25,-485.4 346.94,-485.4 346.94,-521.4" style=""/>
<text text-anchor="middle" x="218.1" y="-499.2" font-family="SimSun" font-size="14.00" style="">session_state 中是否存在 uploaded_content ?</text>
</g>
<!-- check_messages&#45;&gt;check_uploaded_content -->
<g id="edge3" class="edge" data-name="check_messages-&gt;check_uploaded_content">

<path fill="none" stroke="black" d="M231.86,-647.07C245.78,-626.22 264.85,-589.91 255.1,-558.4 252.05,-548.57 246.53,-538.94 240.67,-530.59" style=""/>
<polygon fill="black" stroke="black" points="243.49,-528.52 234.67,-522.66 237.91,-532.74 243.49,-528.52" style=""/>
<text text-anchor="middle" x="263.35" y="-572.2" font-family="Times,serif" font-size="14.00" style="">是</text>
</g>
<!-- create_messages&#45;&gt;check_uploaded_content -->
<g id="edge4" class="edge" data-name="create_messages-&gt;check_uploaded_content">

<path fill="none" stroke="black" d="M170.35,-558.21C177.85,-549.75 187.05,-539.38 195.37,-530.01" style=""/>
<polygon fill="black" stroke="black" points="197.96,-532.36 201.98,-522.56 192.73,-527.72 197.96,-532.36" style=""/>
</g>
<!-- create_uploaded_content -->
<g id="node5" class="node" pointer-events="visible" data-name="create_uploaded_content">

<polygon fill="none" stroke="black" points="262.28,-432.6 11.91,-432.6 11.91,-396.6 262.28,-396.6 262.28,-432.6" style=""/>
<text text-anchor="middle" x="137.1" y="-410.4" font-family="SimSun" font-size="14.00" style="">st.session_state.uploaded_content = None</text>
</g>
<!-- check_uploaded_content&#45;&gt;create_uploaded_content -->
<g id="edge5" class="edge" data-name="check_uploaded_content-&gt;create_uploaded_content">

<path fill="none" stroke="black" d="M202.09,-485.25C190.41,-472.73 174.37,-455.54 161.15,-441.37" style=""/>
<polygon fill="black" stroke="black" points="163.76,-439.04 154.38,-434.12 158.64,-443.82 163.76,-439.04" style=""/>
<text text-anchor="middle" x="190.35" y="-454.8" font-family="Times,serif" font-size="14.00" style="">否</text>
</g>
<!-- check_context_length -->
<g id="node6" class="node" pointer-events="visible" data-name="check_context_length">

<polygon fill="none" stroke="black" points="337.61,-359.6 96.58,-359.6 96.58,-323.6 337.61,-323.6 337.61,-359.6" style=""/>
<text text-anchor="middle" x="217.1" y="-337.4" font-family="SimSun" font-size="14.00" style="">session_state 中是否存在 context_length ?</text>
</g>
<!-- check_uploaded_content&#45;&gt;check_context_length -->
<g id="edge6" class="edge" data-name="check_uploaded_content-&gt;check_context_length">

<path fill="none" stroke="black" d="M236.9,-485.16C257.03,-464.51 284.38,-428.68 271.1,-396.6 266.57,-385.67 258.63,-375.79 250.22,-367.57" style=""/>
<polygon fill="black" stroke="black" points="252.64,-365.04 242.87,-360.93 247.95,-370.24 252.64,-365.04" style=""/>
<text text-anchor="middle" x="280.35" y="-410.4" font-family="Times,serif" font-size="14.00" style="">是</text>
</g>
<!-- create_uploaded_content&#45;&gt;check_context_length -->
<g id="edge7" class="edge" data-name="create_uploaded_content-&gt;check_context_length">

<path fill="none" stroke="black" d="M156.46,-396.41C166.31,-387.67 178.47,-376.88 189.3,-367.27" style=""/>
<polygon fill="black" stroke="black" points="191.34,-370.14 196.5,-360.88 186.7,-364.9 191.34,-370.14" style=""/>
</g>
<!-- create_context_length -->
<g id="node7" class="node" pointer-events="visible" data-name="create_context_length">

<polygon fill="none" stroke="black" points="258.13,-270.8 20.06,-270.8 20.06,-234.8 258.13,-234.8 258.13,-270.8" style=""/>
<text text-anchor="middle" x="139.1" y="-248.6" font-family="SimSun" font-size="14.00" style="">st.session_state.context_length = 12000</text>
</g>
<!-- check_context_length&#45;&gt;create_context_length -->
<g id="edge8" class="edge" data-name="check_context_length-&gt;create_context_length">

<path fill="none" stroke="black" d="M201.69,-323.45C190.44,-310.93 174.99,-293.74 162.26,-279.57" style=""/>
<polygon fill="black" stroke="black" points="165.04,-277.43 155.75,-272.34 159.83,-282.11 165.04,-277.43" style=""/>
<text text-anchor="middle" x="190.35" y="-293" font-family="Times,serif" font-size="14.00" style="">否</text>
</g>
<!-- check_file_content_length -->
<g id="node8" class="node" pointer-events="visible" data-name="check_file_content_length">

<polygon fill="none" stroke="black" points="349.77,-197.8 82.42,-197.8 82.42,-161.8 349.77,-161.8 349.77,-197.8" style=""/>
<text text-anchor="middle" x="216.1" y="-175.6" font-family="SimSun" font-size="14.00" style="">session_state 中是否存在 file_content_length ?</text>
</g>
<!-- check_context_length&#45;&gt;check_file_content_length -->
<g id="edge9" class="edge" data-name="check_context_length-&gt;check_file_content_length">

<path fill="none" stroke="black" d="M234.89,-323.3C253.93,-302.59 279.78,-266.7 267.1,-234.8 262.81,-224.02 255.23,-214.14 247.21,-205.87" style=""/>
<polygon fill="black" stroke="black" points="249.88,-203.58 240.24,-199.2 245.05,-208.64 249.88,-203.58" style=""/>
<text text-anchor="middle" x="276.35" y="-248.6" font-family="Times,serif" font-size="14.00" style="">是</text>
</g>
<!-- create_context_length&#45;&gt;check_file_content_length -->
<g id="edge10" class="edge" data-name="create_context_length-&gt;check_file_content_length">

<path fill="none" stroke="black" d="M157.74,-234.61C167.12,-225.96 178.68,-215.3 189.03,-205.76" style=""/>
<polygon fill="black" stroke="black" points="191.27,-208.45 196.25,-199.1 186.52,-203.31 191.27,-208.45" style=""/>
</g>
<!-- create_file_content_length -->
<g id="node9" class="node" pointer-events="visible" data-name="create_file_content_length">

<polygon fill="none" stroke="black" points="264.29,-109 -0.1,-109 -0.1,-73 264.29,-73 264.29,-109" style=""/>
<text text-anchor="middle" x="132.1" y="-86.8" font-family="SimSun" font-size="14.00" style="">st.session_state.file_content_length = 15000</text>
</g>
<!-- check_file_content_length&#45;&gt;create_file_content_length -->
<g id="edge11" class="edge" data-name="check_file_content_length-&gt;create_file_content_length">

<path fill="none" stroke="black" d="M199.5,-161.65C187.27,-149.01 170.44,-131.62 156.65,-117.37" style=""/>
<polygon fill="black" stroke="black" points="159.47,-115.25 150,-110.5 154.44,-120.12 159.47,-115.25" style=""/>
<text text-anchor="middle" x="187.35" y="-131.2" font-family="Times,serif" font-size="14.00" style="">否</text>
</g>
<!-- end -->
<g id="node10" class="node" pointer-events="visible" data-name="end">

<polygon fill="none" stroke="black" points="243.1,-36 189.1,-36 189.1,0 243.1,0 243.1,-36" style=""/>
<text text-anchor="middle" x="216.1" y="-13.8" font-family="SimSun" font-size="14.00" style="">结束</text>
</g>
<!-- check_file_content_length&#45;&gt;end -->
<g id="edge12" class="edge" data-name="check_file_content_length-&gt;end">

<path fill="none" stroke="black" d="M236.07,-161.5C249.38,-148.57 265.74,-129.63 273.1,-109 278.47,-93.93 279.42,-87.7 273.1,-73 268.28,-61.8 259.87,-51.84 250.97,-43.61" style=""/>
<polygon fill="black" stroke="black" points="253.41,-41.1 243.54,-37.26 248.87,-46.42 253.41,-41.1" style=""/>
<text text-anchor="middle" x="283.35" y="-86.8" font-family="Times,serif" font-size="14.00" style="">是</text>
</g>
<!-- create_file_content_length&#45;&gt;end -->
<g id="edge13" class="edge" data-name="create_file_content_length-&gt;end">

<path fill="none" stroke="black" d="M152.43,-72.81C162.77,-64.07 175.53,-53.28 186.91,-43.67" style=""/>
<polygon fill="black" stroke="black" points="189.12,-46.38 194.5,-37.25 184.61,-41.03 189.12,-46.38" style=""/>
</g>
</g>
</svg>

##### 聊天消息处理流程图设计

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="582pt" height="844pt" viewBox="0.00 0.00 581.70 843.80">
<g id="graph0" class="graph" transform="translate(4,839.7999877929688) scale(1)" data-name="聊天消息处理流程">

<polygon fill="white" stroke="none" points="-4,4 -4,-839.8 577.7,-839.8 577.7,4 -4,4"/>
<!-- start -->
<g id="node1" class="node" pointer-events="visible" data-name="start">

<polygon fill="none" stroke="black" points="512.08,-835.8 412.08,-835.8 412.08,-799.8 512.08,-799.8 512.08,-835.8"/>
<text text-anchor="middle" x="462.08" y="-813.6" font-family="SimSun" font-size="14.00">聊天消息处理部分</text>
</g>
<!-- display_messages -->
<g id="node2" class="node" pointer-events="visible" data-name="display_messages">

<polygon fill="none" stroke="black" points="573.83,-762.6 350.32,-762.6 350.32,-721.4 573.83,-721.4 573.83,-762.6"/>
<text text-anchor="middle" x="462.08" y="-746.2" font-family="SimSun" font-size="14.00">for msg in st.session_state.messages:</text>
<text text-anchor="middle" x="462.08" y="-729.4" font-family="SimSun" font-size="14.00"> 显示消息</text>
</g>
<!-- start&#45;&gt;display_messages -->
<g id="edge1" class="edge" data-name="start-&gt;display_messages">

<path fill="none" stroke="black" d="M462.08,-799.31C462.08,-791.81 462.08,-782.81 462.08,-774.23"/>
<polygon fill="black" stroke="black" points="465.58,-774.48 462.08,-764.48 458.58,-774.48 465.58,-774.48"/>
</g>
<!-- chat_input -->
<g id="node3" class="node" pointer-events="visible" data-name="chat_input">

<polygon fill="none" stroke="black" points="572.22,-684.2 351.93,-684.2 351.93,-648.2 572.22,-648.2 572.22,-684.2"/>
<text text-anchor="middle" x="462.08" y="-662" font-family="SimSun" font-size="14.00">prompt = st.chat_input('请输入问题...')</text>
</g>
<!-- display_messages&#45;&gt;chat_input -->
<g id="edge2" class="edge" data-name="display_messages-&gt;chat_input">

<path fill="none" stroke="black" d="M462.08,-721.22C462.08,-713.45 462.08,-704.36 462.08,-695.89"/>
<polygon fill="black" stroke="black" points="465.58,-696.03 462.08,-686.03 458.58,-696.03 465.58,-696.03"/>
</g>
<!-- check_prompt -->
<g id="node4" class="node" pointer-events="visible" data-name="check_prompt">

<polygon fill="none" stroke="black" points="520.29,-611.2 403.86,-611.2 403.86,-575.2 520.29,-575.2 520.29,-611.2"/>
<text text-anchor="middle" x="462.08" y="-589" font-family="SimSun" font-size="14.00">用户是否输入了问题?</text>
</g>
<!-- chat_input&#45;&gt;check_prompt -->
<g id="edge3" class="edge" data-name="chat_input-&gt;check_prompt">

<path fill="none" stroke="black" d="M462.08,-648.01C462.08,-640.43 462.08,-631.3 462.08,-622.74"/>
<polygon fill="black" stroke="black" points="465.58,-622.74 462.08,-612.74 458.58,-622.74 465.58,-622.74"/>
</g>
<!-- add_user_message -->
<g id="node5" class="node" pointer-events="visible" data-name="add_user_message">

<polygon fill="none" stroke="black" points="523.11,-522.4 223.04,-522.4 223.04,-486.4 523.11,-486.4 523.11,-522.4"/>
<text text-anchor="middle" x="373.08" y="-500.2" font-family="SimSun" font-size="14.00">st.session_state.messages.append({role: 'user', ...})</text>
</g>
<!-- check_prompt&#45;&gt;add_user_message -->
<g id="edge4" class="edge" data-name="check_prompt-&gt;add_user_message">

<path fill="none" stroke="black" d="M444.49,-575.05C431.53,-562.41 413.7,-545.02 399.09,-530.77"/>
<polygon fill="black" stroke="black" points="401.61,-528.34 392.01,-523.87 396.73,-533.35 401.61,-528.34"/>
<text text-anchor="middle" x="430.33" y="-544.6" font-family="Times,serif" font-size="14.00">是</text>
</g>
<!-- end -->
<g id="node13" class="node" pointer-events="visible" data-name="end">

<polygon fill="none" stroke="black" points="460.08,-36 406.08,-36 406.08,0 460.08,0 460.08,-36"/>
<text text-anchor="middle" x="433.08" y="-13.8" font-family="SimSun" font-size="14.00">结束</text>
</g>
<!-- check_prompt&#45;&gt;end -->
<g id="edge5" class="edge" data-name="check_prompt-&gt;end">

<path fill="none" stroke="black" d="M498.27,-574.74C522.92,-560.06 551.08,-536.48 551.08,-505.4 551.08,-505.4 551.08,-505.4 551.08,-90 551.08,-52.19 506.06,-34 471.63,-25.62"/>
<polygon fill="black" stroke="black" points="472.44,-22.22 461.92,-23.47 470.93,-29.05 472.44,-22.22"/>
<text text-anchor="middle" x="556.33" y="-298.6" font-family="Times,serif" font-size="14.00">否</text>
</g>
<!-- display_user_message -->
<g id="node6" class="node" pointer-events="visible" data-name="display_user_message">

<polygon fill="none" stroke="black" points="461.13,-449.2 285.02,-449.2 285.02,-408 461.13,-408 461.13,-449.2"/>
<text text-anchor="middle" x="373.08" y="-432.8" font-family="SimSun" font-size="14.00">with st.chat_message('user'):</text>
<text text-anchor="middle" x="373.08" y="-416" font-family="SimSun" font-size="14.00"> st.markdown(prompt)</text>
</g>
<!-- add_user_message&#45;&gt;display_user_message -->
<g id="edge6" class="edge" data-name="add_user_message-&gt;display_user_message">

<path fill="none" stroke="black" d="M373.08,-485.91C373.08,-478.41 373.08,-469.41 373.08,-460.83"/>
<polygon fill="black" stroke="black" points="376.58,-461.08 373.08,-451.08 369.58,-461.08 376.58,-461.08"/>
</g>
<!-- build_api_request -->
<g id="node7" class="node" pointer-events="visible" data-name="build_api_request">

<polygon fill="none" stroke="black" points="431.38,-370.6 314.77,-370.6 314.77,-329.4 431.38,-329.4 431.38,-370.6"/>
<text text-anchor="middle" x="373.08" y="-354.2" font-family="SimSun" font-size="14.00">构建 API 请求</text>
<text text-anchor="middle" x="373.08" y="-337.4" font-family="SimSun" font-size="14.00">messages_for_api</text>
</g>
<!-- display_user_message&#45;&gt;build_api_request -->
<g id="edge7" class="edge" data-name="display_user_message-&gt;build_api_request">

<path fill="none" stroke="black" d="M373.08,-407.87C373.08,-400.07 373.08,-390.91 373.08,-382.25"/>
<polygon fill="black" stroke="black" points="376.58,-382.44 373.08,-372.44 369.58,-382.44 376.58,-382.44"/>
</g>
<!-- send_api_request -->
<g id="node8" class="node" pointer-events="visible" data-name="send_api_request">

<polygon fill="none" stroke="black" points="506.54,-276.4 239.61,-276.4 239.61,-240.4 506.54,-240.4 506.54,-276.4"/>
<text text-anchor="middle" x="373.08" y="-254.2" font-family="SimSun" font-size="14.00">response = client.chat.completions.create(...)</text>
</g>
<!-- build_api_request&#45;&gt;send_api_request -->
<g id="edge8" class="edge" data-name="build_api_request-&gt;send_api_request">

<path fill="none" stroke="black" d="M373.08,-329.11C373.08,-317.12 373.08,-301.62 373.08,-288.28"/>
<polygon fill="black" stroke="black" points="376.58,-288.32 373.08,-278.32 369.58,-288.32 376.58,-288.32"/>
</g>
<!-- handle_api_response -->
<g id="node9" class="node" pointer-events="visible" data-name="handle_api_response">

<polygon fill="none" stroke="black" points="279.4,-187.4 190.75,-187.4 190.75,-146.2 279.4,-146.2 279.4,-187.4"/>
<text text-anchor="middle" x="235.08" y="-171" font-family="SimSun" font-size="14.00">处理流式回复</text>
<text text-anchor="middle" x="235.08" y="-154.2" font-family="SimSun" font-size="14.00">(显示助手消息)</text>
</g>
<!-- send_api_request&#45;&gt;handle_api_response -->
<g id="edge9" class="edge" data-name="send_api_request-&gt;handle_api_response">

<path fill="none" stroke="black" d="M346.14,-239.91C325.96,-226.81 298.02,-208.67 275.21,-193.86"/>
<polygon fill="black" stroke="black" points="277.35,-191.08 267.06,-188.57 273.54,-196.95 277.35,-191.08"/>
</g>
<!-- api_error -->
<g id="node11" class="node" pointer-events="visible" data-name="api_error">

<polygon fill="none" stroke="black" points="486.2,-184.8 325.95,-184.8 325.95,-148.8 486.2,-148.8 486.2,-184.8"/>
<text text-anchor="middle" x="406.08" y="-162.6" font-family="SimSun" font-size="14.00">是否发生 openai.APIError?</text>
</g>
<!-- send_api_request&#45;&gt;api_error -->
<g id="edge10" class="edge" data-name="send_api_request-&gt;api_error">

<path fill="none" stroke="black" d="M379.44,-240.12C384.09,-227.48 390.51,-210.06 395.88,-195.49"/>
<polygon fill="black" stroke="black" points="399.04,-197.04 399.21,-186.44 392.47,-194.62 399.04,-197.04"/>
<text text-anchor="middle" x="414.6" y="-209.8" font-family="Times,serif" font-size="14.00">API 错误</text>
</g>
<!-- update_messages_assistant -->
<g id="node10" class="node" pointer-events="visible" data-name="update_messages_assistant">

<polygon fill="none" stroke="black" points="324.23,-109 -0.08,-109 -0.08,-73 324.23,-73 324.23,-109"/>
<text text-anchor="middle" x="162.08" y="-86.8" font-family="SimSun" font-size="14.00">st.session_state.messages.append({role: 'assistant', ...})</text>
</g>
<!-- handle_api_response&#45;&gt;update_messages_assistant -->
<g id="edge11" class="edge" data-name="handle_api_response-&gt;update_messages_assistant">

<path fill="none" stroke="black" d="M215.51,-146.02C206.76,-137.17 196.33,-126.63 187.03,-117.22"/>
<polygon fill="black" stroke="black" points="189.79,-115.04 180.27,-110.39 184.81,-119.96 189.79,-115.04"/>
</g>
<!-- update_messages_assistant&#45;&gt;end -->
<g id="edge12" class="edge" data-name="update_messages_assistant-&gt;end">

<path fill="none" stroke="black" d="M228.72,-72.54C280.83,-58.89 351.45,-40.39 394.55,-29.09"/>
<polygon fill="black" stroke="black" points="395.43,-32.48 404.22,-26.56 393.66,-25.71 395.43,-32.48"/>
</g>
<!-- display_api_error_message -->
<g id="node12" class="node" pointer-events="visible" data-name="display_api_error_message">

<polygon fill="none" stroke="black" points="523.52,-109 342.63,-109 342.63,-73 523.52,-73 523.52,-109"/>
<text text-anchor="middle" x="433.08" y="-86.8" font-family="SimSun" font-size="14.00">st.error(error_msg, icon='🚨')</text>
</g>
<!-- api_error&#45;&gt;display_api_error_message -->
<g id="edge13" class="edge" data-name="api_error-&gt;display_api_error_message">

<path fill="none" stroke="black" d="M412.47,-148.31C415.57,-139.85 419.36,-129.5 422.84,-119.98"/>
<polygon fill="black" stroke="black" points="426.05,-121.38 426.2,-110.79 419.48,-118.97 426.05,-121.38"/>
</g>
<!-- display_api_error_message&#45;&gt;end -->
<g id="edge14" class="edge" data-name="display_api_error_message-&gt;end">

<path fill="none" stroke="black" d="M433.08,-72.81C433.08,-65.23 433.08,-56.1 433.08,-47.54"/>
<polygon fill="black" stroke="black" points="436.58,-47.54 433.08,-37.54 429.58,-47.54 436.58,-47.54"/>
</g>
</g>
</svg>

##### 新建对话按钮流程图设计

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="318pt" height="514pt" viewBox="0.00 0.00 317.69 513.60">
<g id="graph0" class="graph" transform="translate(4,509.6000061035156) scale(1)" data-name="新建对话按钮流程">

<polygon fill="white" stroke="none" points="-4,4 -4,-509.6 313.69,-509.6 313.69,4 -4,4"/>
<!-- start -->
<g id="node1" class="node" pointer-events="visible" data-name="start">

<polygon fill="none" stroke="black" points="268.09,-505.6 168.09,-505.6 168.09,-469.6 268.09,-469.6 268.09,-505.6"/>
<text text-anchor="middle" x="218.09" y="-483.4" font-family="SimSun" font-size="14.00">新建对话按钮部分</text>
</g>
<!-- check_new_chat_button -->
<g id="node2" class="node" pointer-events="visible" data-name="check_new_chat_button">

<polygon fill="none" stroke="black" points="309.79,-432.6 126.39,-432.6 126.39,-396.6 309.79,-396.6 309.79,-432.6"/>
<text text-anchor="middle" x="218.09" y="-410.4" font-family="SimSun" font-size="14.00">st.button('新建对话') 是否被点击?</text>
</g>
<!-- start&#45;&gt;check_new_chat_button -->
<g id="edge1" class="edge" data-name="start-&gt;check_new_chat_button">

<path fill="none" stroke="black" d="M218.09,-469.41C218.09,-461.83 218.09,-452.7 218.09,-444.14"/>
<polygon fill="black" stroke="black" points="221.59,-444.14 218.09,-434.14 214.59,-444.14 221.59,-444.14"/>
</g>
<!-- reset_messages -->
<g id="node3" class="node" pointer-events="visible" data-name="reset_messages">

<polygon fill="none" stroke="black" points="242.13,-343.8 60.06,-343.8 60.06,-307.8 242.13,-307.8 242.13,-343.8"/>
<text text-anchor="middle" x="151.09" y="-321.6" font-family="SimSun" font-size="14.00">st.session_state.messages = []</text>
</g>
<!-- check_new_chat_button&#45;&gt;reset_messages -->
<g id="edge2" class="edge" data-name="check_new_chat_button-&gt;reset_messages">

<path fill="none" stroke="black" d="M204.86,-396.45C195.28,-384.05 182.17,-367.07 171.3,-352.98"/>
<polygon fill="black" stroke="black" points="174.34,-351.19 165.46,-345.41 168.8,-355.46 174.34,-351.19"/>
<text text-anchor="middle" x="206.84" y="-366" font-family="Times,serif" font-size="14.00">被点击</text>
</g>
<!-- end -->
<g id="node7" class="node" pointer-events="visible" data-name="end">

<polygon fill="none" stroke="black" points="247.09,-36 193.09,-36 193.09,0 247.09,0 247.09,-36"/>
<text text-anchor="middle" x="220.09" y="-13.8" font-family="SimSun" font-size="14.00">结束</text>
</g>
<!-- check_new_chat_button&#45;&gt;end -->
<g id="edge3" class="edge" data-name="check_new_chat_button-&gt;end">

<path fill="none" stroke="black" d="M239.42,-396.36C256.66,-380.36 278.09,-354.83 278.09,-326.8 278.09,-326.8 278.09,-326.8 278.09,-90 278.09,-71.98 266.75,-56.02 254.1,-43.81"/>
<polygon fill="black" stroke="black" points="256.63,-41.38 246.83,-37.35 251.98,-46.61 256.63,-41.38"/>
<text text-anchor="middle" x="293.84" y="-204.2" font-family="Times,serif" font-size="14.00">未点击</text>
</g>
<!-- reset_uploaded_content -->
<g id="node4" class="node" pointer-events="visible" data-name="reset_uploaded_content">

<polygon fill="none" stroke="black" points="250.28,-270.8 -0.09,-270.8 -0.09,-234.8 250.28,-234.8 250.28,-270.8"/>
<text text-anchor="middle" x="125.09" y="-248.6" font-family="SimSun" font-size="14.00">st.session_state.uploaded_content = None</text>
</g>
<!-- reset_messages&#45;&gt;reset_uploaded_content -->
<g id="edge4" class="edge" data-name="reset_messages-&gt;reset_uploaded_content">

<path fill="none" stroke="black" d="M144.8,-307.61C141.92,-299.76 138.45,-290.27 135.22,-281.46"/>
<polygon fill="black" stroke="black" points="138.57,-280.44 131.85,-272.25 132,-282.84 138.57,-280.44"/>
</g>
<!-- reset_current_file -->
<g id="node5" class="node" pointer-events="visible" data-name="reset_current_file">

<polygon fill="none" stroke="black" points="246.05,-182 30.13,-182 30.13,-146 246.05,-146 246.05,-182"/>
<text text-anchor="middle" x="138.09" y="-159.8" font-family="SimSun" font-size="14.00">st.session_state.current_file = None</text>
</g>
<!-- reset_uploaded_content&#45;&gt;reset_current_file -->
<g id="edge5" class="edge" data-name="reset_uploaded_content-&gt;reset_current_file">

<path fill="none" stroke="black" d="M127.66,-234.65C129.41,-222.96 131.77,-207.19 133.81,-193.62"/>
<polygon fill="black" stroke="black" points="137.24,-194.32 135.26,-183.91 130.32,-193.28 137.24,-194.32"/>
</g>
<!-- success_new_chat -->
<g id="node6" class="node" pointer-events="visible" data-name="success_new_chat">

<polygon fill="none" stroke="black" points="241.29,-109 82.9,-109 82.9,-73 241.29,-73 241.29,-109"/>
<text text-anchor="middle" x="162.09" y="-86.8" font-family="SimSun" font-size="14.00">st.success('新对话已创建！')</text>
</g>
<!-- reset_current_file&#45;&gt;success_new_chat -->
<g id="edge6" class="edge" data-name="reset_current_file-&gt;success_new_chat">

<path fill="none" stroke="black" d="M143.9,-145.81C146.56,-137.96 149.76,-128.47 152.74,-119.66"/>
<polygon fill="black" stroke="black" points="155.97,-121.06 155.85,-110.46 149.33,-118.82 155.97,-121.06"/>
</g>
<!-- success_new_chat&#45;&gt;end -->
<g id="edge7" class="edge" data-name="success_new_chat-&gt;end">

<path fill="none" stroke="black" d="M176.13,-72.81C182.97,-64.44 191.34,-54.2 198.93,-44.9"/>
<polygon fill="black" stroke="black" points="201.61,-47.16 205.22,-37.2 196.19,-42.73 201.61,-47.16"/>
</g>
</g>
</svg>

##### 文件上传与处理流程设计

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="483pt" height="862pt" viewBox="0.00 0.00 482.86 862.00">
<g id="graph0" class="graph" transform="translate(4,858) scale(1)" data-name="文件上传与处理流程">

<polygon fill="white" stroke="none" points="-4,4 -4,-858 478.86,-858 478.86,4 -4,4"/>
<!-- start -->
<g id="node1" class="node" pointer-events="visible" data-name="start">

<polygon fill="none" stroke="black" points="406.78,-854 327.78,-854 327.78,-818 406.78,-818 406.78,-854"/>
<text text-anchor="middle" x="367.28" y="-831.8" font-family="SimSun" font-size="14.00">文件上传部分</text>
</g>
<!-- file_uploader -->
<g id="node2" class="node" pointer-events="visible" data-name="file_uploader">

<polygon fill="none" stroke="black" points="474.94,-781 259.61,-781 259.61,-745 474.94,-745 474.94,-781"/>
<text text-anchor="middle" x="367.28" y="-758.8" font-family="SimSun" font-size="14.00">uploaded_file = st.file_uploader(...)</text>
</g>
<!-- start&#45;&gt;file_uploader -->
<g id="edge1" class="edge" data-name="start-&gt;file_uploader">

<path fill="none" stroke="black" d="M367.28,-817.81C367.28,-810.23 367.28,-801.1 367.28,-792.54"/>
<polygon fill="black" stroke="black" points="370.78,-792.54 367.28,-782.54 363.78,-792.54 370.78,-792.54"/>
</g>
<!-- check_file_uploaded -->
<g id="node3" class="node" pointer-events="visible" data-name="check_file_uploaded">

<polygon fill="none" stroke="black" points="414.99,-708 319.57,-708 319.57,-672 414.99,-672 414.99,-708"/>
<text text-anchor="middle" x="367.28" y="-685.8" font-family="SimSun" font-size="14.00">是否上传了文件?</text>
</g>
<!-- file_uploader&#45;&gt;check_file_uploaded -->
<g id="edge2" class="edge" data-name="file_uploader-&gt;check_file_uploaded">

<path fill="none" stroke="black" d="M367.28,-744.81C367.28,-737.23 367.28,-728.1 367.28,-719.54"/>
<polygon fill="black" stroke="black" points="370.78,-719.54 367.28,-709.54 363.78,-719.54 370.78,-719.54"/>
</g>
<!-- check_file_changed -->
<g id="node4" class="node" pointer-events="visible" data-name="check_file_changed">

<polygon fill="none" stroke="black" points="392.49,-619.2 234.07,-619.2 234.07,-583.2 392.49,-583.2 392.49,-619.2"/>
<text text-anchor="middle" x="313.28" y="-597" font-family="SimSun" font-size="14.00">上传文件与当前文件是否不同?</text>
</g>
<!-- check_file_uploaded&#45;&gt;check_file_changed -->
<g id="edge3" class="edge" data-name="check_file_uploaded-&gt;check_file_changed">

<path fill="none" stroke="black" d="M356.61,-671.85C349.04,-659.69 338.73,-643.11 330.07,-629.18"/>
<polygon fill="black" stroke="black" points="333.16,-627.54 324.91,-620.9 327.22,-631.24 333.16,-627.54"/>
<text text-anchor="middle" x="349.53" y="-641.4" font-family="Times,serif" font-size="14.00">是</text>
</g>
<!-- end -->
<g id="node11" class="node" pointer-events="visible" data-name="end">

<polygon fill="none" stroke="black" points="377.28,-36 323.28,-36 323.28,0 377.28,0 377.28,-36"/>
<text text-anchor="middle" x="350.28" y="-13.8" font-family="SimSun" font-size="14.00">结束</text>
</g>
<!-- check_file_uploaded&#45;&gt;end -->
<g id="edge4" class="edge" data-name="check_file_uploaded-&gt;end">

<path fill="none" stroke="black" d="M385.82,-671.59C401.1,-655.28 420.28,-629.39 420.28,-602.2 420.28,-602.2 420.28,-602.2 420.28,-90 420.28,-68.53 403.85,-51.2 386.91,-39.03"/>
<polygon fill="black" stroke="black" points="389.07,-36.27 378.8,-33.65 385.2,-42.1 389.07,-36.27"/>
<text text-anchor="middle" x="425.53" y="-343.6" font-family="Times,serif" font-size="14.00">否</text>
</g>
<!-- process_file -->
<g id="node5" class="node" pointer-events="visible" data-name="process_file">

<polygon fill="none" stroke="black" points="346.34,-530.4 0.22,-530.4 0.22,-494.4 346.34,-494.4 346.34,-530.4"/>
<text text-anchor="middle" x="173.28" y="-508.2" font-family="SimSun" font-size="14.00">processed_content = process_uploaded_file(uploaded_file)</text>
</g>
<!-- check_file_changed&#45;&gt;process_file -->
<g id="edge5" class="edge" data-name="check_file_changed-&gt;process_file">

<path fill="none" stroke="black" d="M285.28,-582.84C263.92,-569.6 234.27,-551.21 210.84,-536.69"/>
<polygon fill="black" stroke="black" points="212.86,-533.82 202.51,-531.52 209.17,-539.77 212.86,-533.82"/>
<text text-anchor="middle" x="261.53" y="-552.6" font-family="Times,serif" font-size="14.00">是</text>
</g>
<!-- check_file_changed&#45;&gt;end -->
<g id="edge6" class="edge" data-name="check_file_changed-&gt;end">

<path fill="none" stroke="black" d="M335.3,-582.72C352.77,-566.75 374.28,-541.4 374.28,-513.4 374.28,-513.4 374.28,-513.4 374.28,-90 374.28,-75.21 369.55,-59.55 364.28,-46.71"/>
<polygon fill="black" stroke="black" points="367.51,-45.35 360.24,-37.63 361.11,-48.19 367.51,-45.35"/>
<text text-anchor="middle" x="379.53" y="-286.8" font-family="Times,serif" font-size="14.00">否</text>
</g>
<!-- check_processed_content -->
<g id="node6" class="node" pointer-events="visible" data-name="check_processed_content">

<polygon fill="none" stroke="black" points="226.49,-457.4 120.07,-457.4 120.07,-421.4 226.49,-421.4 226.49,-457.4"/>
<text text-anchor="middle" x="173.28" y="-435.2" font-family="SimSun" font-size="14.00">文件处理是否成功?</text>
</g>
<!-- process_file&#45;&gt;check_processed_content -->
<g id="edge7" class="edge" data-name="process_file-&gt;check_processed_content">

<path fill="none" stroke="black" d="M173.28,-494.21C173.28,-486.63 173.28,-477.5 173.28,-468.94"/>
<polygon fill="black" stroke="black" points="176.78,-468.94 173.28,-458.94 169.78,-468.94 176.78,-468.94"/>
</g>
<!-- update_session_state -->
<g id="node7" class="node" pointer-events="visible" data-name="update_session_state">

<polygon fill="none" stroke="black" points="256.92,-368.4 59.63,-368.4 59.63,-327.2 256.92,-327.2 256.92,-368.4"/>
<text text-anchor="middle" x="158.28" y="-352" font-family="SimSun" font-size="14.00">更新 session_state</text>
<text text-anchor="middle" x="158.28" y="-335.2" font-family="SimSun" font-size="14.00">(uploaded_content, current_file)</text>
</g>
<!-- check_processed_content&#45;&gt;update_session_state -->
<g id="edge8" class="edge" data-name="check_processed_content-&gt;update_session_state">

<path fill="none" stroke="black" d="M170.39,-421.12C168.43,-409.44 165.79,-393.69 163.48,-379.86"/>
<polygon fill="black" stroke="black" points="166.99,-379.62 161.88,-370.33 160.08,-380.77 166.99,-379.62"/>
<text text-anchor="middle" x="172.53" y="-390.8" font-family="Times,serif" font-size="14.00">是</text>
</g>
<!-- error_message_process -->
<g id="node12" class="node" pointer-events="visible" data-name="error_message_process">

<polygon fill="none" stroke="black" points="351.19,-309 207.37,-309 207.37,-273 351.19,-273 351.19,-309"/>
<text text-anchor="middle" x="279.28" y="-286.8" font-family="SimSun" font-size="14.00">st.error('文件处理失败...')</text>
</g>
<!-- check_processed_content&#45;&gt;error_message_process -->
<g id="edge9" class="edge" data-name="check_processed_content-&gt;error_message_process">

<path fill="none" stroke="black" d="M207.84,-421.15C228.11,-409.2 252.41,-391.35 266.28,-368.6 274.98,-354.33 278.3,-335.94 279.42,-320.75"/>
<polygon fill="black" stroke="black" points="282.91,-321.05 279.83,-310.91 275.91,-320.75 282.91,-321.05"/>
<text text-anchor="middle" x="256.53" y="-390.8" font-family="Times,serif" font-size="14.00">否</text>
</g>
<!-- generate_system_prompt -->
<g id="node8" class="node" pointer-events="visible" data-name="generate_system_prompt">

<polygon fill="none" stroke="black" points="224.78,-255 135.78,-255 135.78,-219 224.78,-219 224.78,-255"/>
<text text-anchor="middle" x="180.28" y="-232.8" font-family="SimSun" font-size="14.00">生成系统提示词</text>
</g>
<!-- update_session_state&#45;&gt;generate_system_prompt -->
<g id="edge10" class="edge" data-name="update_session_state-&gt;generate_system_prompt">

<path fill="none" stroke="black" d="M162.31,-326.84C165.73,-309.96 170.67,-285.54 174.5,-266.58"/>
<polygon fill="black" stroke="black" points="177.93,-267.3 176.48,-256.8 171.06,-265.91 177.93,-267.3"/>
</g>
<!-- update_messages_with_system_prompt -->
<g id="node9" class="node" pointer-events="visible" data-name="update_messages_with_system_prompt">

<polygon fill="none" stroke="black" points="291.82,-182 98.74,-182 98.74,-146 291.82,-146 291.82,-182"/>
<text text-anchor="middle" x="195.28" y="-159.8" font-family="SimSun" font-size="14.00">st.session_state.messages = [...]</text>
</g>
<!-- generate_system_prompt&#45;&gt;update_messages_with_system_prompt -->
<g id="edge11" class="edge" data-name="generate_system_prompt-&gt;update_messages_with_system_prompt">

<path fill="none" stroke="black" d="M183.91,-218.81C185.53,-211.14 187.49,-201.89 189.31,-193.24"/>
<polygon fill="black" stroke="black" points="192.73,-194.02 191.37,-183.51 185.88,-192.57 192.73,-194.02"/>
</g>
<!-- success_message -->
<g id="node10" class="node" pointer-events="visible" data-name="success_message">

<polygon fill="none" stroke="black" points="294.97,-109 125.59,-109 125.59,-73 294.97,-73 294.97,-109"/>
<text text-anchor="middle" x="210.28" y="-86.8" font-family="SimSun" font-size="14.00">st.success('文档...解析完成！')</text>
</g>
<!-- update_messages_with_system_prompt&#45;&gt;success_message -->
<g id="edge12" class="edge" data-name="update_messages_with_system_prompt-&gt;success_message">

<path fill="none" stroke="black" d="M198.91,-145.81C200.53,-138.14 202.49,-128.89 204.31,-120.24"/>
<polygon fill="black" stroke="black" points="207.73,-121.02 206.37,-110.51 200.88,-119.57 207.73,-121.02"/>
</g>
<!-- success_message&#45;&gt;end -->
<g id="edge13" class="edge" data-name="success_message-&gt;end">

<path fill="none" stroke="black" d="M244.53,-72.63C265.16,-62.17 291.47,-48.83 312.68,-38.07"/>
<polygon fill="black" stroke="black" points="314.26,-41.19 321.59,-33.55 311.09,-34.95 314.26,-41.19"/>
</g>
<!-- error_message_process&#45;&gt;end -->
<g id="edge14" class="edge" data-name="error_message_process-&gt;end">

<path fill="none" stroke="black" d="M296.88,-272.65C311.97,-255.86 331.28,-228.92 331.28,-201 331.28,-201 331.28,-201 331.28,-90 331.28,-75.49 335.03,-59.87 339.19,-46.98"/>
<polygon fill="black" stroke="black" points="342.38,-48.48 342.38,-37.88 335.77,-46.16 342.38,-48.48"/>
</g>
</g>
</svg>

##### 侧边栏交互流程设计

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="166pt" height="620pt" viewBox="0.00 0.00 166.31 620.00">
<g id="graph0" class="graph" transform="translate(4,616) scale(1)" data-name="侧边栏交互流程">

<polygon fill="white" stroke="none" points="-4,4 -4,-616 162.31,-616 162.31,4 -4,4"/>
<!-- sidebar_start -->
<g id="node1" class="node" pointer-events="visible" data-name="sidebar_start">

<polygon fill="none" stroke="black" points="129.7,-612 28.61,-612 28.61,-576 129.7,-576 129.7,-612"/>
<text text-anchor="middle" x="79.16" y="-589.8" font-family="SimSun" font-size="14.00">with st.sidebar:</text>
</g>
<!-- config_header -->
<g id="node2" class="node" pointer-events="visible" data-name="config_header">

<polygon fill="none" stroke="black" points="140.17,-540 18.14,-540 18.14,-504 140.17,-504 140.17,-540"/>
<text text-anchor="middle" x="79.16" y="-517.8" font-family="SimSun" font-size="14.00">st.header('配置参数')</text>
</g>
<!-- sidebar_start&#45;&gt;config_header -->
<g id="edge1" class="edge" data-name="sidebar_start-&gt;config_header">

<path fill="none" stroke="black" d="M79.16,-575.7C79.16,-568.41 79.16,-559.73 79.16,-551.54"/>
<polygon fill="black" stroke="black" points="82.66,-551.62 79.16,-541.62 75.66,-551.62 82.66,-551.62"/>
</g>
<!-- model_selectbox -->
<g id="node3" class="node" pointer-events="visible" data-name="model_selectbox">

<polygon fill="none" stroke="black" points="148.23,-468 10.08,-468 10.08,-432 148.23,-432 148.23,-468"/>
<text text-anchor="middle" x="79.16" y="-445.8" font-family="SimSun" font-size="14.00">st.selectbox('选择模型')</text>
</g>
<!-- config_header&#45;&gt;model_selectbox -->
<g id="edge2" class="edge" data-name="config_header-&gt;model_selectbox">

<path fill="none" stroke="black" d="M79.16,-503.7C79.16,-496.41 79.16,-487.73 79.16,-479.54"/>
<polygon fill="black" stroke="black" points="82.66,-479.62 79.16,-469.62 75.66,-479.62 82.66,-479.62"/>
</g>
<!-- temperature_slider -->
<g id="node4" class="node" pointer-events="visible" data-name="temperature_slider">

<polygon fill="none" stroke="black" points="137.47,-396 20.84,-396 20.84,-360 137.47,-360 137.47,-396"/>
<text text-anchor="middle" x="79.16" y="-373.8" font-family="SimSun" font-size="14.00">st.slider('温度参数')</text>
</g>
<!-- model_selectbox&#45;&gt;temperature_slider -->
<g id="edge3" class="edge" data-name="model_selectbox-&gt;temperature_slider">

<path fill="none" stroke="black" d="M79.16,-431.7C79.16,-424.41 79.16,-415.73 79.16,-407.54"/>
<polygon fill="black" stroke="black" points="82.66,-407.62 79.16,-397.62 75.66,-407.62 82.66,-407.62"/>
</g>
<!-- max_tokens_slider -->
<g id="node5" class="node" pointer-events="visible" data-name="max_tokens_slider">

<polygon fill="none" stroke="black" points="137.47,-324 20.84,-324 20.84,-288 137.47,-288 137.47,-324"/>
<text text-anchor="middle" x="79.16" y="-301.8" font-family="SimSun" font-size="14.00">st.slider('最大长度')</text>
</g>
<!-- temperature_slider&#45;&gt;max_tokens_slider -->
<g id="edge4" class="edge" data-name="temperature_slider-&gt;max_tokens_slider">

<path fill="none" stroke="black" d="M79.16,-359.7C79.16,-352.41 79.16,-343.73 79.16,-335.54"/>
<polygon fill="black" stroke="black" points="82.66,-335.62 79.16,-325.62 75.66,-335.62 82.66,-335.62"/>
</g>
<!-- context_length_slider -->
<g id="node6" class="node" pointer-events="visible" data-name="context_length_slider">

<polygon fill="none" stroke="black" points="142.47,-252 15.84,-252 15.84,-216 142.47,-216 142.47,-252"/>
<text text-anchor="middle" x="79.16" y="-229.8" font-family="SimSun" font-size="14.00">st.slider('上下文长度')</text>
</g>
<!-- max_tokens_slider&#45;&gt;context_length_slider -->
<g id="edge5" class="edge" data-name="max_tokens_slider-&gt;context_length_slider">

<path fill="none" stroke="black" d="M79.16,-287.7C79.16,-280.41 79.16,-271.73 79.16,-263.54"/>
<polygon fill="black" stroke="black" points="82.66,-263.62 79.16,-253.62 75.66,-263.62 82.66,-263.62"/>
</g>
<!-- file_content_length_slider -->
<g id="node7" class="node" pointer-events="visible" data-name="file_content_length_slider">

<polygon fill="none" stroke="black" points="158.47,-180 -0.16,-180 -0.16,-144 158.47,-144 158.47,-180"/>
<text text-anchor="middle" x="79.16" y="-157.8" font-family="SimSun" font-size="14.00">st.slider('文件内容读取长度')</text>
</g>
<!-- context_length_slider&#45;&gt;file_content_length_slider -->
<g id="edge6" class="edge" data-name="context_length_slider-&gt;file_content_length_slider">

<path fill="none" stroke="black" d="M79.16,-215.7C79.16,-208.41 79.16,-199.73 79.16,-191.54"/>
<polygon fill="black" stroke="black" points="82.66,-191.62 79.16,-181.62 75.66,-191.62 82.66,-191.62"/>
</g>
<!-- new_chat_button -->
<g id="node8" class="node" pointer-events="visible" data-name="new_chat_button">

<polygon fill="none" stroke="black" points="139.64,-108 18.67,-108 18.67,-72 139.64,-72 139.64,-108"/>
<text text-anchor="middle" x="79.16" y="-85.8" font-family="SimSun" font-size="14.00">st.button('新建对话')</text>
</g>
<!-- file_content_length_slider&#45;&gt;new_chat_button -->
<g id="edge7" class="edge" data-name="file_content_length_slider-&gt;new_chat_button">

<path fill="none" stroke="black" d="M79.16,-143.7C79.16,-136.41 79.16,-127.73 79.16,-119.54"/>
<polygon fill="black" stroke="black" points="82.66,-119.62 79.16,-109.62 75.66,-119.62 82.66,-119.62"/>
</g>
<!-- sidebar_end -->
<g id="node9" class="node" pointer-events="visible" data-name="sidebar_end">

<polygon fill="none" stroke="black" points="113.16,-36 45.15,-36 45.15,0 113.16,0 113.16,-36"/>
<text text-anchor="middle" x="79.16" y="-13.8" font-family="SimSun" font-size="14.00">侧边栏结束</text>
</g>
<!-- new_chat_button&#45;&gt;sidebar_end -->
<g id="edge8" class="edge" data-name="new_chat_button-&gt;sidebar_end">

<path fill="none" stroke="black" d="M79.16,-71.7C79.16,-64.41 79.16,-55.73 79.16,-47.54"/>
<polygon fill="black" stroke="black" points="82.66,-47.62 79.16,-37.62 75.66,-47.62 82.66,-47.62"/>
</g>
</g>
</svg>

#### 关键模块

- **文件处理** (`process_uploaded_file`)
  - TXT文件：直接读取UTF-8编码
  - PDF文件：使用PyMuPDF解析文本
  - 内容截取：保留前N个字符（N=context_length）

- **消息管理**
  - 保留最近10条消息（5轮对话）
  - 系统消息处理规则：
    - 文件更新时添加文档摘要
    - 文件移除时提示模式切换

- **API调用**

  ```python
  response = client.chat.completions.create(
      model=model_list[selected_model],
      messages=messages_for_api,  # 包含系统提示的完整上下文
      stream=True,                  # 启用流式传输
      temperature=temperature,
      max_tokens=8192              # 固定最大输出长度
  )
  ```

#### 注意事项

10. API密钥需通过`.env`文件配置
11. PDF解析依赖字体嵌入，可能影响格式还原
12. 上下文长度设置需考虑模型实际支持的最大长度
13. 温度值设置低于0.3时可能产生确定性响应
14. 文件上传大小受Streamlit默认限制（200MB）

#### 错误处理机制

- 文件解析异常：显示红色错误提示
- API调用失败：回滚最近消息记录
- 流式中断：保留已接收内容
- 类型错误：自动过滤无效消息类型

#### 交互优化

- 采用流式响应
- 上下文窗口滑动管理
- 文件内容预处理器
- 对话缓存策略（保留最近5轮）

#### 性能优化建议

- 单次对话上下文建议 ≤16000字符
- PDF文件页数建议 ≤50页
- 复杂问题建议拆分为多步骤提问

___
PS: 如果是你，你会如何实现本次的实战项目代码逻辑，你觉得有更好的实现方案吗？
