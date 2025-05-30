import torch
import clip


label_text_map = []

with open('text/ntu120_label_map.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        label_text_map.append(line.rstrip().lstrip())
# 读取ntu120的label


paste_text_map0 = []

with open('text/synonym_openai_t01.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(',')
        paste_text_map0.append(temp_list)
# 读取每个label对应的近义词，通过位置来对应

paste_text_map1 = []

with open('text/sentence_openai_t01.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split('.')
        while len(temp_list) < 4:
            temp_list.append(" ")
        paste_text_map1.append(temp_list)
# 读取每个label对应的描述句子，通过位置来对应，填充长度为4

paste_text_map2 = []

with open('text/pasta_openai_t01.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        paste_text_map2.append(temp_list)
# 读取每个label对应骨架六个部分的描述句子，通过位置来对应

ucla_paste_text_map0 = []

with open('text/ucla_synonym_openai_t01.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(',')
        ucla_paste_text_map0.append(temp_list)


ucla_paste_text_map1 = []

with open('text/ucla_pasta_openai_t01.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        ucla_paste_text_map1.append(temp_list)




def text_prompt():
    '''
        classes:拼接后的文本编码
        num_text_aug:模板数量
        text_dict:存储每个模板生成编码的字典
    '''
    text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]
    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for c in label_text_map])


    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug,text_dict


def text_prompt_openai_random():
    '''
        total_list:返回每个label对应的近义词的clip编码
    '''
    print("Use text prompt openai synonym random")

    total_list = []
    for pasta_list in paste_text_map0:
        temp_list = []
        for item in pasta_list:
            temp_list.append(clip.tokenize(item))
        total_list.append(temp_list)
    return total_list

def text_prompt_openai_random_bert():
    '''
        total_list:返回每个label对应的近义词列表
    '''
    print("Use text prompt openai synonym random bert")
    
    total_list = []
    for pasta_list in paste_text_map0:
        temp_list = []
        for item in pasta_list:
            temp_list.append(item)
        total_list.append(temp_list)
    return total_list



def text_prompt_openai_pasta_pool_4part():
    '''
        classes:拼接后的文本编码,label加对应部位描述一起送入clip处理后得到向量,再将不同类的相同部位向量cat起来
        num_text_aug:模板数量
        text_dict:存储每个模板生成编码的字典
    '''
    print("Use text prompt openai pasta pool")
    text_dict = {}
    num_text_aug = 5

    for ii in range(num_text_aug):
        if ii == 0:
            #  paste_text_map2为[[label, 六个部位描述]. [].  ....]
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in paste_text_map2])
        elif ii == 1:
            text_dict[ii] = torch.cat([clip.tokenize((','.join(pasta_list[0:2]))) for pasta_list in paste_text_map2])
        elif ii == 2:
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','.join(pasta_list[2:4]))) for pasta_list in paste_text_map2])
        elif ii == 3:
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','+ pasta_list[4])) for pasta_list in paste_text_map2])
        else:
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0]+','+','.join(pasta_list[5:]))) for pasta_list in paste_text_map2])


    classes = torch.cat([v for k, v in text_dict.items()])
    
    return classes, num_text_aug, text_dict

def text_prompt_openai_pasta_pool_4part_bert():
    print("Use text prompt openai pasta pool bert")
    text_dict = {}
    num_text_aug = 5

    for ii in range(num_text_aug):
        if ii == 0:
            input_list = [pasta_list[ii] for pasta_list in paste_text_map2]
            text_dict[ii] = input_list
        elif ii == 1:
            input_list = [','.join(pasta_list[0:2]) for pasta_list in paste_text_map2]
            text_dict[ii] = input_list
        elif ii == 2:
            input_list = [pasta_list[0] +','.join(pasta_list[2:4]) for pasta_list in paste_text_map2]
            text_dict[ii] = input_list
        elif ii == 3:
            input_list = [pasta_list[0] +','+ pasta_list[4] for pasta_list in paste_text_map2]
            text_dict[ii] = input_list
        else:
            input_list = [pasta_list[0]+','+','.join(pasta_list[5:]) for pasta_list in paste_text_map2]
            text_dict[ii] = input_list

    
    return num_text_aug, text_dict



def text_prompt_openai_random_ucla():
    print("Use text prompt openai synonym random UCLA")

    total_list = []
    for pasta_list in ucla_paste_text_map0:
        temp_list = []
        for item in pasta_list:
            temp_list.append(clip.tokenize(item))
        total_list.append(temp_list)
    return total_list


def text_prompt_openai_pasta_pool_4part_ucla():
    print("Use text prompt openai pasta pool ucla")
    text_dict = {}
    num_text_aug = 5

    for ii in range(num_text_aug):
        if ii == 0:
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in ucla_paste_text_map1])
        elif ii == 1:
            text_dict[ii] = torch.cat([clip.tokenize((','.join(pasta_list[0:2]))) for pasta_list in ucla_paste_text_map1])
        elif ii == 2:
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','.join(pasta_list[2:4]))) for pasta_list in ucla_paste_text_map1])
        elif ii == 3:
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','+ pasta_list[4])) for pasta_list in ucla_paste_text_map1])
        else:
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0]+','+','.join(pasta_list[5:]))) for pasta_list in ucla_paste_text_map1])



    classes = torch.cat([v for k, v in text_dict.items()])
    
    return classes, num_text_aug, text_dict

