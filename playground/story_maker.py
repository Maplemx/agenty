import os
import Agently

agent_factory = Agently.AgentFactory()


agent_factory\
    .set_settings("model_settings.model_name", "OpenAI")\
    .set_settings("model_settings.auth", { "api_key": "YOUR-OPENAI-API-KEY" })\
    .set_settings("model_settings.url", "YOUR-BASE-URL-IF-NEEDED")

'''
agent_factory\
    .set_settings("model_settings.model_name", "ERNIE")\
    .set_settings("model_settings.auth", { "aistudio": "YOUR-BAIDU-AISTUDIO-ACCESS-TOKEN" })
'''

# Step 1: 准备Agent
director = agent_factory.create_agent("agently_story_maker")

# Step 2: 询问用户设想
print("[让我们开始构想吧]")
choice = None

## 角色创作
existed_characters = director.agent_storage.table("story").get("characters", [])
if len(existed_characters) > 0:
    print("已经存在之前创作过的角色...")
    for character in existed_characters:
        print(character)
    while choice not in ("1", "2", "3"):
        choice = str(input("请选择接下来的操作：[1]直接使用这些角色 [2]添加新的角色 [3]清空这些角色，重新创作角色: "))
else:
    choice = 2

is_finish = None
if choice == "1":
    is_finish = True
if choice == "3":
    director.agent_storage.table("story").set("characters", []).save()

while not is_finish:
    role_desc = {}
    confirm = False
    while not confirm:
        role_desc["name"] = str(input("请输入角色名称: "))
        role_desc["desc"] = str(input("简单描述一下这个角色: "))
        role_desc["extra_rules"] = str(input("还有什么需要补充的要求、规则吗？（没有可留空）: "))
        print("[人物设定]\n", f"姓名: { role_desc['name'] }\n", f"简介: { role_desc['desc'] }\n", f"其他要求: { role_desc['extra_rules'] }")
        while confirm not in ("y", "n"):
            confirm = str(input("请确认信息是否正确[Y/N]: ")).lower()
        confirm = True if confirm == "y" else False
    print("[现在正在为该人物生成小传，请稍候...]")
    confirm = False
    last_version = None
    edit_suggestion = None
    while not confirm:
        input_dict = {
            "角色名称": role_desc["name"],
            "角色简述": role_desc["desc"],
            "其他要求": role_desc["extra_rules"],
        }
        if last_version:
            input_dict["上一个版本"] = last_version
        if edit_suggestion:
            input_dict["修改建议"] = edit_suggestion
        character_setting = director\
            .input(input_dict)\
            .instruct("根据{input}提供的信息，生成符合{output}要求的角色设定信息")\
            .output({
                "姓名": ("String", "必须使用{角色名称}"),
                "年龄": ("Number", "角色年龄"),
                "性格特点": ("String", "该角色的性格特点和行为癖好"),
                "背景故事": [("String", "用年份表的方式生成角色的背景故事资料")],
            })\
            .on_delta(lambda data: print(data, end=""))\
            .start()
        while confirm not in ("y", "n"):
            confirm = str(input("\n是否满意这个人物小传[Y/N]: ")).lower()
        if confirm == "y":
            confirm = True
        else:
            last_version = character_setting
            edit_suggestion = str(input("请输入修改建议（没有可留空）: "))
            confirm = False
    director.agent_storage.table("story").append("characters", character_setting).save()
    print("[人物信息已经保存]")
    is_finish = None
    while is_finish not in ("y", "n"):
        is_finish = str(input("请问是否继续设定新的角色[Y/N]:")).lower()
    is_finish = True if is_finish == 'n' else False

## 剧情概要设想
story_guide = str(input("请输入你希望演出的剧情概要: "))

# Step 3: 演出开始
print("[现在演出开始]")
is_finish = None
lines_count = 0
history = []
edit_suggestion = None
last_version = None
character_settings = director.agent_storage.table("story").get("characters")
while not is_finish:
    print("[正在生成剧情分支，请稍候...]")
    next_act_input_dict = {
        "story_guide": story_guide,
        "history": history,
        "character_settings": character_settings,
    }
    instruction = ["根据{input.story_guide}提供的信息，在{output.next_act}中遵照给定的输出格式，生成最多3个可能的剧情走向分支"]
    if edit_suggestion != None:
        next_act_input_dict["edit_suggestion"] = edit_suggestion
        instruction.append("请遵照{input.edit_suggestion}的建议，修正{last_version}的内容或重新生成新的内容")
    if last_version != None:
        next_act_input_dict["last_version"] = last_version
    next_act = director\
        .input(next_act_input_dict)\
        .instruct(instruction)\
        .output({
            "next_act": [{
                "next_story_desc": ("String", "根据{input.story_guide}和已经发生过的剧情{chat_history}简述你认为接下来的情节发展"),
                "next_lines": [("String", "从{character_settings}中选择行动的角色，输出“{name}：”作为开头，然后写出{name}接下来的台词或行动，使用\`<具体台词>\`的格式，行动如对谁说话，走向哪里，使用(<具体行动>)的格式")],
            }],
        })\
        .start()
    next_lines = {}
    for index, content in enumerate(next_act["next_act"]):
        next_lines[str(index + 1)] = "\n".join(content["next_lines"])
        print(f"【分支[{ index + 1 }]】\n", f"[故事走向]{content['next_story_desc']}\n", f"[具体台词]\n{ next_lines[str(index + 1)] }\n")
    user_selection = None
    while user_selection == None:
        user_selection = str(input("请选出你喜欢的故事走向，输入序号（都不喜欢输入0）："))
        if user_selection not in next_lines and user_selection != "0":
            user_selection = None
    if user_selection == "0":
        edit_suggestion = str(input("请输入你的修改意见: "))
        last_version = next_lines
    else:
        edit_suggestion = None
        last_version = None
        history.append(next_lines[user_selection])
    is_finish = None
    while is_finish == None:
        is_finish = str(input("请问是否继续生成[Y/N]:")).lower()
        is_finish = True if is_finish == 'n' else False

confirm = None
while confirm not in ("y", "n"):
    confirm = str(input("请问是否需要保存这段剧情[Y/N]: ")).lower()
if confirm == "y":
    file_name = str(input("请输入希望保存的文件名: "))
    if not file_name.endswith(".txt"):
        file_name += ".txt"
    with open(f"./{ file_name }", "w") as file:
        log = "\n".join(history)
        file.write(log)
        print("[保存完毕，感谢您的使用，再见👋]")