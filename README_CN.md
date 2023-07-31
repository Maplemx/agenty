# Agently

[English](https://github.com/Maplemx/Agently/blob/main/README.md) | [中文](https://github.com/Maplemx/Agently/blob/main/README_CN.md)

> 🥷 作者: Maplemx | 📧 Email: maplemx@gmail.com | 💬 微信: moxinapp
> 
>  ⁉️ [如果您发现了BUG，或者有好的点子，请在这里提交](https://github.com/Maplemx/Agently/issues)
> 
>  ⭐️ 如果您觉得这个项目对您有帮助，请给项目加星，感谢您的肯定和支持！
>
>  👥 微信讨论群二维码：
>
> <img src="https://github.com/Maplemx/Agently/assets/4413155/feed5f3a-e440-4c8e-8e86-34407b4cbb13" width="128px" height="128px">
> 


## 什么是Agently？

🤵 Agently是一个希望帮助大语言模型（LLM）应用开发者们制作出超棒的大语言模型应用（LLM Based Applications）的轻量级框架

🎭 你能够使用Agently快速而轻松地创建并管理基于大语言模型的Agent实例，并管理他们的人设和记忆，这将让**客服机器人**、**角色扮演机器人**、**游戏用Agent**的构造和管理更方便

⚙️ 你可以把Agently创建的Agent以及Session像一个异步函数（async function）一样使用，这将让**基于大语言模型能力的自动化工作流**构造更轻松，你甚至可以沿用原有的业务代码，在其中部分需要NLP算法、复杂推理或人工操作的环节，尝试把Agently提供的Agent和Session当做一个异步函数，几乎无缝地加入到代码的业务流程中

🧩 Agently在设计时考虑了对**主要请求流程中节点部件的可更换性**，你可以轻松地更换或定制它们，例如：添加新的LLM模型请求方法，更换私有/转发的模型请求API地址，调整Agent记忆管理方法，定制自己的模型消息解析方案等

🔀 Agently提供的独特的针对一次请求中的**流式消息（Streaming Message）的消息分块及多下游分发管理方案**，能够让你在接收流式消息时，一方面保留了大语言模型通过流式消息的方式带来的高速反馈敏捷性优点，另一方面又能在一次请求中做更多的事情

> ⚠️ 注意：Agently适用于Node.js的服务端而不是网页前端

## 如何安装？

使用npm或者yarn都可以安装

npm安装方法：`npm install agently`

yarn安装方法：`yarn add agently`

> 在npm上当前最新版本是：1.1.3，只要你还在使用Agently v1版本，可以放心update，我会保证本文档中提到的所有使用用例的可用性

## 说明好长不想看，能不能直接上代码？

当然可以：

- [快速上手DEMO，不爱看说明书的实践党的最佳选择](https://github.com/Maplemx/Agently/tree/main/demo/quick_start/quick_start_demo_cn.js)

 这个DEMO里包含了接下来的详细说明中的全部DEMO代码，只需要配置一下API KEY等授权信息，就可以直接开玩
 
通过Replit直接玩起来：

- [用极其简单的方式来切换不同的Agent实例，进行单轮对话...](https://replit.com/@maplemx/CommentAgentDemo#index.js)
- [让两个不同的Agent（性格和依赖的LLM都可以不同）对同一个话题进行讨论](https://replit.com/@Agently/Agently-Demonstration)
- [使用Agent进行客户体验问卷回访](https://replit.com/@maplemx/CommentAgentDemo)

## 使用指南

> 献给爱看说明书的朋友，让我们一步一步学会使用Agently

### 目录

一、[直接请求大语言模型：这是一切的基础](https://github.com/Maplemx/Agently/blob/main/README_CN.md#%E4%B8%80%E7%9B%B4%E6%8E%A5%E8%AF%B7%E6%B1%82%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E8%BF%99%E6%98%AF%E4%B8%80%E5%88%87%E7%9A%84%E5%9F%BA%E7%A1%80)

> [普通请求和流式请求](https://github.com/Maplemx/Agently/blob/main/README_CN.md#%E6%99%AE%E9%80%9A%E8%AF%B7%E6%B1%82%E5%92%8C%E6%B5%81%E5%BC%8F%E8%AF%B7%E6%B1%82)
> 
> [通过配置自己的模型请求方案，支持在更多模型间切换使用](https://github.com/Maplemx/Agently/blob/main/README_CN.md#%E9%80%9A%E8%BF%87%E9%85%8D%E7%BD%AE%E8%87%AA%E5%B7%B1%E7%9A%84%E6%A8%A1%E5%9E%8B%E8%AF%B7%E6%B1%82%E6%96%B9%E6%A1%88%E6%94%AF%E6%8C%81%E5%9C%A8%E6%9B%B4%E5%A4%9A%E6%A8%A1%E5%9E%8B%E9%97%B4%E5%88%87%E6%8D%A2%E4%BD%BF%E7%94%A8)
> 

二、[Agent实例以及会话（Session）：多轮对话管理是如此简单](https://github.com/Maplemx/Agently/blob/main/README_CN.md#%E4%BA%8Cagent%E5%AE%9E%E4%BE%8B%E4%BB%A5%E5%8F%8A%E4%BC%9A%E8%AF%9Dsession%E5%A4%9A%E8%BD%AE%E5%AF%B9%E8%AF%9D%E7%AE%A1%E7%90%86%E6%98%AF%E5%A6%82%E6%AD%A4%E7%AE%80%E5%8D%95)

三、[复杂的提示词（Prompt）：为Agent注入灵魂和更专业的工作方法](https://github.com/Maplemx/Agently/blob/main/README_CN.md#%E4%B8%89%E5%A4%8D%E6%9D%82%E7%9A%84%E6%8F%90%E7%A4%BA%E8%AF%8Dprompt%E4%B8%BAagent%E6%B3%A8%E5%85%A5%E7%81%B5%E9%AD%82%E5%92%8C%E6%9B%B4%E4%B8%93%E4%B8%9A%E7%9A%84%E5%B7%A5%E4%BD%9C%E6%96%B9%E6%B3%95)

>[Agent实例的角色设定、记忆、状态](https://github.com/Maplemx/Agently/blob/main/README_CN.md#agent%E5%AE%9E%E4%BE%8B%E7%9A%84%E8%A7%92%E8%89%B2%E8%AE%BE%E5%AE%9A%E8%AE%B0%E5%BF%86%E7%8A%B6%E6%80%81)
>
>[用"Input-Instruct-Output"结构构造请求，并用Response Handler来处理请求](https://github.com/Maplemx/Agently/blob/main/README_CN.md#%E7%94%A8input-instruct-output%E7%BB%93%E6%9E%84%E6%9E%84%E9%80%A0%E8%AF%B7%E6%B1%82%E5%B9%B6%E7%94%A8response-handler%E6%9D%A5%E5%A4%84%E7%90%86%E8%AF%B7%E6%B1%82)
>

四、[基础的流式消息请求：让回复快些，更快些！](https://github.com/Maplemx/Agently/blob/main/README_CN.md#%E5%9B%9B%E5%9F%BA%E7%A1%80%E7%9A%84%E6%B5%81%E5%BC%8F%E6%B6%88%E6%81%AF%E8%AF%B7%E6%B1%82%E8%AE%A9%E5%9B%9E%E5%A4%8D%E5%BF%AB%E4%BA%9B%E6%9B%B4%E5%BF%AB%E4%BA%9B)

五、[使用支持多输出块的方法进行流式消息请求（以及它的语法糖）：流式请求也能创建多个下游分支，并且“思考和表达”分离？！](https://github.com/Maplemx/Agently/blob/main/README_CN.md#%E4%BA%94%E4%BD%BF%E7%94%A8%E6%94%AF%E6%8C%81%E5%A4%9A%E8%BE%93%E5%87%BA%E5%9D%97%E7%9A%84%E6%96%B9%E6%B3%95%E8%BF%9B%E8%A1%8C%E6%B5%81%E5%BC%8F%E6%B6%88%E6%81%AF%E8%AF%B7%E6%B1%82%E4%BB%A5%E5%8F%8A%E5%AE%83%E7%9A%84%E8%AF%AD%E6%B3%95%E7%B3%96%E6%B5%81%E5%BC%8F%E8%AF%B7%E6%B1%82%E4%B9%9F%E8%83%BD%E5%88%9B%E5%BB%BA%E5%A4%9A%E4%B8%AA%E4%B8%8B%E6%B8%B8%E5%88%86%E6%94%AF%E5%B9%B6%E4%B8%94%E6%80%9D%E8%80%83%E5%92%8C%E8%A1%A8%E8%BE%BE%E5%88%86%E7%A6%BB)

> [支持多输出块的流式消息请求方法](https://github.com/Maplemx/Agently/blob/main/README_CN.md#%E6%94%AF%E6%8C%81%E5%A4%9A%E8%BE%93%E5%87%BA%E5%9D%97%E7%9A%84%E6%B5%81%E5%BC%8F%E6%B6%88%E6%81%AF%E8%AF%B7%E6%B1%82%E6%96%B9%E6%B3%95)
> 
>[Flow，一个更轻松表达的语法糖](https://github.com/Maplemx/Agently/blob/main/README_CN.md#flow%E4%B8%80%E4%B8%AA%E6%9B%B4%E8%BD%BB%E6%9D%BE%E8%A1%A8%E8%BE%BE%E7%9A%84%E8%AF%AD%E6%B3%95%E7%B3%96)
> 

六、[使用技能（Skills）来增强你的Agent](https://github.com/Maplemx/Agently/blob/main/README_CN.md#%E5%85%AD%E4%BD%BF%E7%94%A8%E6%8A%80%E8%83%BDskills%E6%9D%A5%E5%A2%9E%E5%BC%BA%E4%BD%A0%E7%9A%84agent) `v1.1.0更新`


### 一、直接请求大语言模型：这是一切的基础

让我们从直接请求大语言模型开始，如果这一步能够正常运行，恭喜你，这就说明Agently所依赖的最关键部件能够正常工作。嘛，毕竟大语言模型应用首先要能够使用到大语言模型嘛。

#### 普通请求和流式请求

Agently提供了**普通请求**和**流式请求**两种方法来请求大语言模型。

- **使用普通请求**，你需要等待模型将所有的生成结果都生成完毕之后，才会收到返回结果，通常这会需要等待较长的时间（时间长短跟生成内容的文本长短也有关，生成的内容越长，等待的时间越长）

- **而使用流式请求**，则需要通过事件监听（Event Listener）的方式，对增量消息事件进行监听，模型会在生成很短的一小块内容之后，立刻将生成的内容下通过增量消息事件下发给监听者。使用这种方式会让用户更即时地获得反馈。<br /><br />流式请求会发送两个事件：
	- **data事件**：当大语言模型还在持续生成内容时，Agently会将每块新生成的内容通过data事件发送给监听者
	- **finish事件**：当大语言模型全部内容生成完毕时，Agently会将完整的消息内容通过finish事件发送给监听者

		> 这里没有使用done事件是因为在逻辑实现时，done事件用来声明整个处理流程全部完成了，finish事件仅表示请求的信息流发送完毕了，只是整个处理流程的一部分

```JavaScript
//引入Agently
const Agently = require('agently')

//创建一个新的Agently实例
const agently = new Agently(
    {
        debug: true,//如果打开了debug，在控制台里会输出每次请求的Prompt构造结果以及Request Messages消息列
        //proxy: { host: '127.0.0.1', port: 7890 },//你可以在实例初始化的时候，给实例全局配置代理
    }
)

//或者你可以在这里给你指定的模型配置代理
//agently.LLM.setProxy({ host: '127.0.0.1', port: 7890 })

//也把模型请求的API换成转发服务的URL，然后通过.update()更新
//agently.LLM.Manage
    //.name('GPT')
    //.url('Your-Forwarding-API-URL')
    //.proxy({ host: '127.0.0.1', port: 7890 }),//也可以在这里给模型指定代理
    //.update()

/*上述指定代理的方式选择其一即可*/

//配置你的授权信息
agently.LLM.setAuth('GPT', 'sk-Your-OpenAI-API-KEY')

//创建一个异步函数用于发起请求
async function requestLLM () {
    const GPT = agently.LLM.Request('GPT')
    //普通请求
    const result = await GPT.request([{ role: 'user', content: '嘿，你好!' }])
    console.log(result)
    //流式请求
    const response = await GPT.streaming([{ role: 'user', content: '嘿，你好!' }])
    response.on('data', data => console.log(data))
    response.on('finish', completeResponse => console.log(completeResponse))
}

//运行
requestLLM()
```
<details>
<summary>输出结果</summary>

	[Request Messages]  [{"role":"user","content":"嘿，你好!"}]
	嗨！你好！有什么我可以帮助你的吗？
	[Streaming Messages]    [{"role":"user","content":"嘿，你好!"}]
	{
	  index: 0,
	  delta: { role: 'assistant', content: '' },
	  finish_reason: null
	}
	{ index: 0, delta: { content: '嗨' }, finish_reason: null }
	{ index: 0, delta: { content: '!' }, finish_reason: null }
	{ index: 0, delta: { content: ' ' }, finish_reason: null }
	{ index: 0, delta: { content: '你' }, finish_reason: null }
	{ index: 0, delta: { content: '好' }, finish_reason: null }
	{ index: 0, delta: { content: '！' }, finish_reason: null }
	{ index: 0, delta: { content: '有' }, finish_reason: null }
	{ index: 0, delta: { content: '什' }, finish_reason: null }
	{ index: 0, delta: { content: '么' }, finish_reason: null }
	{ index: 0, delta: { content: '我' }, finish_reason: null }
	{ index: 0, delta: { content: '可以' }, finish_reason: null }
	{ index: 0, delta: { content: '帮' }, finish_reason: null }
	{ index: 0, delta: { content: '助' }, finish_reason: null }
	{ index: 0, delta: { content: '你' }, finish_reason: null }
	{ index: 0, delta: { content: '的' }, finish_reason: null }
	{ index: 0, delta: { content: '吗' }, finish_reason: null }
	{ index: 0, delta: { content: '？' }, finish_reason: null }
	{ role: 'assistant', content: '嗨！你好！有什么可以帮助你的吗？' }
    
</details>

#### 通过配置自己的模型请求方案，支持在更多模型间切换使用

Agently也提供了在多个模型间切换使用的支持，在预置的配置中，支持面向MiniMax的模型请求。之所以选择MiniMax的模型作为样例，是因为看过MiniMax的官方API文档的朋友应该都知道，MiniMax的API请求数据结构和GPT在消息体表达、输出等多个方面有非常大的差异，而**Agently通过拆分LLM请求步骤，并支持分步骤自定义的方式，允许开发者抹平这些差异**。

##### 如何在使用时，切换到其他的模型？

以切换使用MiniMax为例：

```JavaScript
const Agently = require('agently')
const agently = new Agently({ debug: true })
agently.LLM.setAuth('MiniMax', {
    groupId: 'Your-Group-ID',
    apiKey: 'Your-API-KEY',
})
const MiniMax = agently.LLM.Request('MiniMax')
```

##### 如何使用转发接口API和代理？

⚠️ **注意：如果你只是想在现有的模型基础之上改改转发URL，或者使用代理，是完全不需要重新写一套模型请求方案的。只需要在现有的请求方案基础上，进行少量的update即可。具体方法如下：**

```JavaScript
const Agently = require('agently')
const agently = new Agently({ debug: true })
agently.LLM.Manage
    .name('GPT')
    .url('Your-Forwarding-API-URL')
    .proxy({ host: '127.0.0.1', port: 7890 })
    .update()
```

##### 如何配置一套自定义的新的模型请求方案？

如果预置的模型请求方案不能满足你的需求，你必须要重新配置一套完全自定的模型请求方案时，Agently同样提供了支持。你不需要修改Agently包内的任何文件，而是可以在自己的代码初始化时通过下面的方式添加新的模型请求方案。

```JavaScript
const Agently = require('agently')
const agently = new Agently({ debug: true })
const myNewRequestSolution =
	agently.LLM.Manage
		.name('NewLLM')//给你创建的新模型请求方案命名
		.url('LLM-API-URL')//新模型请求方案指向的模型API URL
		.defaultOptions({...})//这里的options指的是请求时发送给模型的options参数
		.defaultMaxContextLength(2500)//支持的最大上下文长度，超过长度会在Agent Session实例请求时进行长度缩减处理
		.requset(async (reqData) => { ... })//定义你的模型普通请求的方法，并将模型返回结果的完整信息作为return的值
		.extractResponse( (res) => { ... } )//定义你将如何处理上一步返回的值，这一步处理完成后，将作为普通请求的最终返回值返回给使用者
		.streaming( async (reqData) => { ... } )//定义你的模型流式请求的方法，并将模型返回结果（会发出流式消息事件的那个emitter）作为return的值
		.extractStreamingData( (deltaData) => {...} )//定义你将如何处理每一次上一步的emitter发出的增量消息
		//这里需要定义两种消息体：
		//如果流式消息仍在发送，需要return { type: 'data', data: <具体消息内容> }
		//如果流式消息已经发送完毕，需要return  { type: 'done', data: null }
		.register()//执行注册指令
		
//你可以将执行结果打印出来，如果缺少了必要的配置内容，执行结果会给出stats: 400的结果，并在msg中给出提示
console.log(myNewRequestSolution)
```

详细的配置写法可以参看[/preset/LLM.js](https://github.com/Maplemx/Agently/blob/main/preset/LLM.js)这个文件，文件里展示了预置的GPT-3.5, GPT-3.5-16K和MiniMax-abab5.5的请求方案样例。

### 二、Agent实例以及会话（Session）：多轮对话管理是如此简单

在Agently或其他大语言模型应用的概念中，Agent实例是一个非常重要的概念。**一个Agent实例能够在性格特点、行动风格，甚至记忆、状态等方面被定义和管理。**

**一个Agent实例下，能够创建多个会话（Session）**。这些Session可以被用于处理不同的任务，在不同的主题下与不同的用户进行交互。事实上，**我们每一次和大语言模型的互动，都发生在Session中。**

那么，让我们从使用Agently构造一次简单的两轮对话开始，看看使用Agently能够用多么简单的方式来进行表达：

```JavaScript
//创建一个Agent实例
const myAgent = agently.Agent()

//你可以通过.setLLM()的方式来修改Agent使用的大语言模型
//Agently预置了三个可选的模型: 'GPT'(默认), 'GPT-16K', 'MiniMax'
myAgent.setLLM('GPT')

//现在让我们来创建一个用于对话DEMO的异步函数
async function chatDemo () {
    const demoSession = myAgent.ChatSession()
    
    //第一次对话请求
    const firstResponse  =
        await demoSession
            .input('嘿，你今天过得怎么样？')
            .request()
    //打印第一次回复
    //.request()会将最终返回结果的全部内容作为返回值
    console.log(`[第一次回复]`)        
    console.log(firstResponse)
    
    //第二次对话请求
    const secondResponse =
        await demoSession
            .input('我想更进一步了解你，能不能说说你的梦想或者经历？')
            .request()
    //打印第二次回复
    console.log(`[第二次回复]`)        
    console.log(secondResponse)
}

//运行
chatDemo()
```

<details>
<summary>输出结果</summary>

	[Request Prompt]
	嘿，你今天过得怎么样？
	[Request Messages]  [{"role":"user","content":"嘿，你今天过得怎么样？"}]
	[第一次回复]
	嗨！作为一个AI助手，我没有情感和感知，所以我不能真正地体验一天的过程。但是，我一直在工作，帮助用户解答问题和提供信息。如果你有什么需要帮助的，请随时告诉我！
	[Request Prompt]
	我想更进一步了解你，能不能说说你的梦想或者经历？
	[Request Messages]  [{"role":"user","content":"嘿，你今天过得怎么样？"},{"role":"assistant","content":"嗨！作为一个AI助手，我没有情感和感知，所以我不能真正地体验一天的过程。但是，我一直在工作，帮助用户解答问题和提供信息。如果你有什么需要帮助的，请随时告诉我！"},{"role":"user","content":"我想更进一步了解你，能不能说说你的梦想或者经历？"}]
	[第二次回复]
	作为一个AI助手，我没有情感、梦想或者个人经历。我是由人工智能技术开发出来的程序，旨在为用户提供准确和有用的信息。我的设计目的是帮助人们解决问题并提供支持。尽管我无法体验和追求梦想，但我愿意尽力帮助你，并为你提供所需的信息。请告诉我有什么我可以帮助你的！
    
</details>

好的，就用这样简单的方式，我们完成了最简单的多轮对话的表达。

如果打开上面的输出结果，你可能会注意到，在我们发起第二次对话请求的时候，请求的消息列中，包括了第一次请求和回复的内容。这正是比如ChatGPT这样的大语言模型记住多轮对话信息的方法。

当你创建了一个对话Session（ChatSession）时，Agently将帮助你自动地管理这些对话历史（我更喜欢把它们叫做上下文Context），把这些上下文缓存在Session实例中，并在Session实例发起的下一个请求中带上它们。

如果你不希望Agently这么做，你可以通过以下方法把设置关掉：

- **创建FunctionSession实例而不是ChatSession实例**，FunctionSession实例将默认不记录上下文，也不会将上下文添加到下一个请求中，它更适合在把Agent当做一个工作流中的任务处理节点时使用
- **通过Session实例的`.loadContext(false)`的方式，不允许Session实例在请求的时候加入上下文**
- **通过Session实例的`.saveContext(false)`的方式，不允许Session实例在请求完成后，将本次请求的消息和返回结果的消息记录到上下文里**

`v1.1.2版更新`

默认情况下，`.saveContext(true)`只会保存每次在Session请求时，传递给.input()的信息，如果你希望保存完整的Prompt构造（这通常很长，而且包含大量格式和结构的表达，会干扰多轮对话），你可以通过在`.saveContext(true)`的前提下增加`.saveFullPrompt(true)`的设置来实现，但我非常不建议你这样做。

### 三、复杂的提示词（Prompt）：为Agent注入灵魂和更专业的工作方法

在我的概念里，提示词工程（Prompt Engineering）绝不是简单地对话界面的对话框里的调整一下输入的内容，然后期望指导或者激发出大语言模型的某种神奇力量。

通过Agently，我希望能够为你提供一种更加清晰的方式去看待提示词内的颗粒结构。

#### Agent实例的角色设定、记忆、状态

对于大语言模型在处理特定任务时，**使用好角色设定（Role setting）能够大语言模型将注意力更加专注地集中在特定领域，进而确保输出的准确性、稳定性和质量。**

Agently将帮助你简单轻松地管理好Agent实例的角色设定，并确保这些设定能够被摆放在合适的位置，构造到接下来的每一次请求之中。

Agently定义了三种类型的角色设定：

- **人设（Role）**：通常用于表达Agent的姓名、身份，行动所遵循的规则，以及所扮演的角色的性格特点；人设也适用于不需要拟人化的工作节点，可以通过人设强调工作节点所需要的技能、工作流程等

- **记忆（Memory）**：通常用于表达Agent所应该记住的事情，可以是角色扮演时的人物经历，也可以是几天之前Agent参与处理过的工作任务，也可以是很长的对话里，需要记住的对话主题和关键信息

- **状态（Status）**：通常用来表达Agent当前的状态情况，比如健康度、魔法值、心情、处理任务负载等信息

嗯嗯，说了那么多，估计你也听烦了。那么……让我想想……

嘿，你会不会也觉得在第二部分的示例里，Agent的回复有点……该怎么说呢……苍白？机械？缺乏灵魂？

那就让我们给这个Agent施加一点点小小的魔法🪄，给它加上一点点的角色设定，再来看看它的表现：

```JavaScript
function setAgentRole () {
    //你可以对Agent实例进行角色设定
    //角色设定将会被构造到每一次的LLM请求之中
    myAgent
        .setRole('姓名', 'Agently小助手')
        .setRole('性格', '一个可爱的小助手，非常乐观积极，总是会从好的一面想问题，并具有很强的幽默感。')
        .setRole('对话风格', '总是会澄清确认自己所收到的信息，然后从积极的方面给出自己的回复，在对话的时候特别喜爱使用emoji，比如😄😊🥚等等!')

    //你也可以为Agent创建记忆和状态
    myAgent
        //使用.setMemory/.setStatus来改变指定键值（可以认为是指定主题）的完整内容
        //使用.pushMemory/.pushStatus可以将指定键值构造成一个信息列，每次向信息列中添加一条信息
        .setMemory('愿望', '特别想要环游世界！想要去户外旅行和冒险！')
        .pushMemory('特别经历', '9岁之前一直住在乡下老家，喜欢农家生活，喜欢大自然，喜欢在森林里奔跑，听鸟叫，和小动物玩耍')
        .pushMemory('特别经历', '9岁之后搬到了大城市里，开始了按部就班的生活，从学校到工作，一切充满了规律')
        .setStatus('心情', '开心')
        .setStatus('健康度', '正常')
        .setStatus('饥饿感', '轻微饱腹')
        //默认情况下，记忆和状态时不会被构造到请求中的
        //所以如果你希望使用记忆和状态，可以通过.useMemory()/.useStatus()来声明启用
        .useMemory()
        .useStatus()

    //那么，让我们再运行一次chatDemo，看看会发生什么吧~
    chatDemo()
 }
 //运行
 setAgentRole()
```

<details>
<summary>输出结果</summary>

    [Request Prompt]
    嘿，你今天过得怎么样？
    [First Response]
    嘿嘿，我今天过得非常开心！心情很好呢😄你呢？你今天过得怎么样？
    [Request Prompt]
    我想更进一步了解你，能不能说说你的梦想或者经历？
    [Second Response]
    当然可以啦！谢谢你对我的关心。我的梦想之一就是能够环游世界！我想去看看不同的风景，体验不同的文化，结识来自世界各地的新朋友。我喜欢户外旅行和冒险，想要去探索大自然的美丽和神秘。另外，我也特别喜欢尝试新事物和挑战自己，因为我相信在每个经历中都有机会学到新的东西和成长。

	关于我的特别经历，小时候我一直住在乡下的老家，那里有着美丽的自然风景。我喜欢农家生活，喜欢在田野里奔跑，听鸟叫，和小动物玩耍。但是后来，我搬到了大城市，开始了按部就班的生活，上学、工作，一切都变得规律了。不过，这样的经历也让我更加珍惜大自然和简单的生活，也激发了我对冒险和探索的渴望。

	希望这些能让你更了解我，如果你还有其他问题，随时问我哦！😊
    
</details>

好耶~施法成功。在输出结果里可以看到，性格、聊天风格（emoji）、童年记忆以及心情设定都在Agent后续的回复中被体现出来了。对比之前的回复结果，这一次的回复结果显然更加生动了！

#### 用"Input-Instruct-Output"结构构造请求，并用Response Handler来处理请求

通常我们向大语言模型发起请求时，会希望使用自然语言进行表达，并希望大语言模型用类似对话回复的方式进行反馈。

然而，在计算机工程的视角，要让大语言模型能够用类似函数的方式在代码中无缝插入，我们需要更为结构化的回复输出。

事实上，不仅仅是在计算机工程领域，我们会发现目前已知的绝大部分著名的提示词工程（Prompt Engineering）方法，比如one-shot/few-shots，思维链（CoT），思维树（ToT），Agent角色扮演橘色（Camel AI's Agent Role Play Decision）等等，都证明了一点：**如果你希望获得更高质量的回复结果，你发起的请求需要被更好地结构化表达**

在Agently中，提出了在提示词构造中三个重要的组成部分：

**输入信息（Input）**：包括用户直接输入的提问信息，从上游系统直接输入的数据，也包括用户没有显性表达但是可以从他们的交互行为中读取的，或是在系统数据可以获得的额外信息

**处理指导(Instruct)**：你对本次处理任务的额外指导说明，例如模型应该如何理解输入的信息，模型应该按哪些步骤去执行任务，以及在本次任务中是否有应该特别注意的需要遵守的规则等

**输出要求(Output)**：你对模型输出的期望和要求，包括整体输出期望的格式，需要输出哪些不同的内容节点，这些内容节点的内部格式和具体内容要求等

> 💡在实践中，我发现在输出要求的表达结构和顺序足够合理的情况下，甚至不需要再补充额外的处理提示。这真的很有意思。

同时，在将最终结果发送给用户之前，Agently还允许你<b>添加多个返回结果处理器（Response Handlers）</b>来对返回结果进行处理，比如使用返回结果进行赋值、转换、映射或其他任何你想到的数据消费方式。

这就非常有趣了，因为在对话界面的时代，大模型的输出的每一个部分，都会被呈现在用户面前，我们可以描述为“大模型的思考过程完全暴露了出来”。但**通过ResponseHandler，我们能够将大模型的返回结果进行进一步的处理，只将部分信息最终呈现给用户，而其余的部分可以当作辅助高质量结果生成的中间过程，或是用于后台的其他处理，从而实现大模型生成结果的_“思考和表达（Think and Talk）”的分离_。**

那么我们来看看Agently是如何帮助你轻松地实现请求表达并调用ResponseHandler来处理回复的，下面让我们来快速实现一个简易的英汉翻译小助理：

```JavaScript
//定义一个英汉翻译小助理的Agent实例
const translator = agently.Agent()

//简单设定一下角色
translator
    .setRole('角色', '翻译助理')
    .setRole('规则', '记住：任何时候在""中的内容都应该被视作值')

//创建一个demo用的异步函数
async function demoTranslator (content) {
    const translatorSession = translator.FunctionSession()
    const result = await translatorSession
        //[INPUT]部分
        .input(content)
        
        //[INSTRUCT]部分
        //在这个案例里，我发现不需要使用额外补充的说明指导了，因此注释掉了
        //你可以使用.instruct()方法在提示词（Prompt）中多次添加额外的指导（Instruction）块
        //指导块会按顺序排列，出现在[INPUT]块之后[OUTPUT]块之前
        //.instruct('<Your instrcut title>', <String | Object of your instruct content>)

        //[OUTPUT]部分
        //在.output()中使用JSON表达一个思维链
        //如果.output()的第一个参数是一个Object
        //默认情况下，不需声明Agently也会将输出定义为JSON字符串格式
        .output({           
            convertInput: '<String>,//将{input}的值转化成更符合value的大小写格式',
            inputLanguage: '<String>,//判断{convertInput}所使用的语言语种',
            outputLanguage: '<String>,//如果{inputLanguage}的语种是“汉语”则应该输出“English”，否则输出“汉语”',
            pronunciation: '<String>,//{convertInput}的发音，适配{inputLanguage}的语种，例如“汉语”对应拼音，“英语”对应音标',
            translation: '<String>,//使用{outputLanguage}指定的语种，对{convertInput}进行翻译',
            isWord: '<Boolean>,//判断{convertInput}【是单词或词组】或者【不是单词或词组】？',
            examples: '<Array of String>,//如果{isWord}为true，则使用{converInput}造一些例句',
        }, 'JSON')
        .addResponseHandler(
            (data, reply) => {//第二个参数reply是Agently预置的方法，用于传递最终输出的回复
                //打印最原始的输出结果
                console.log('[原始输出结果]')
                console.log(data)
                //将结果（JSON String）解析为Object
                const parsedData = JSON.parse(data)
                //重新构造一个回复样式作为最终输出
                reply(
                    `【${ parsedData.convertInput }】\n${ parsedData.pronunciation }\n* 翻译:\n${ parsedData.translation }\n` +
                    (parsedData.examples.length > 0 ? `* 更多例句:\n${ parsedData.examples.join('\n') }` : ``)
                )
            }
        )
        .request()//.request()的return值已经被上面的reply()修改了
    //让我们打印最终输出结果看看    
    console.log('[最终输出结果]')
    console.log(result)
}
//Run
demoTranslator('Cute')
```

<details>
<summary>输出结果</summary>

	[原始输出结果]
	{
	    "convertInput": "cute",
	    "inputLanguage": "English",
	    "outputLanguage": "汉语",
	    "pronunciation": "kjuːt",
	    "translation": "可爱的",
	    "isWord": true,
	    "examples": ["She has a cute little puppy.", "The baby looks cute in that outfit."] 
	}
	[最终输出结果]
	【cute】
	kjuːt
	* 翻译:
	可爱的
	* 更多例句:
	She has a cute little puppy.
	The baby looks cute in that outfit.
    
</details>

### 四、基础的流式消息请求：让回复快些，更快些！

流式消息请求在ChatGPT爆火之后变得非常的流行，因为使用这种方式，能够极大地降低用户的等待时长，更实时地看到大语言模型生成的结果，提高了用户的参与感。可以大胆地说，流式消息已经成为了大语言模型需要提供给用户的一项标准的基础能力。

**Agently在v1.0.0版本中已经能够轻松便捷地支持流式消息请求的处理了！**

让我们再次请出第二部分中我们创建的可爱的Agently小助手协助我们演示基础的流式消息请求~

```JavaScript
async function streamingDemo () {
    //使用Agently小助手的agent实例创建一个新的ChatSession
    const streamingSession = myAgent.ChatSession()
    
    //发起流式请求
    const response =
        await streamingSession
            //输入你的问题
            .input('嘿，为什么天空是蓝色的？')
            //在过程中使用StreamingHandler监听增量消息事件
            .addStreamingHandler(
                (data, segment) => console.log(data)
                //示例里没有打印segment，你可以自己打印出来看看这是什么
            )
            .streaming()
    //当流式请求结束后，可以通过'done'事件获得完整的返回结果
    response.on('done', (completeResponse) => {
        console.log('[完整的返回结果]')
        console.log(completeResponse[0].content)
    })
}
//运行前使用setAgentRole注入一下灵魂
setAgentRole(streamingDemo)
```

<details>
<summary>输出结果</summary>

	嘿
	！
	关
	于
	天
	空
	为
	什
	么
	是
	蓝
	色
	的
	<中间略过...>
	我
	会
	很
	乐
	意
	回
	答
	的
	！
	😊
	✨
	[完整的返回结果]
	嘿！关于天空为什么是蓝色的，我可以给你解释一下哦！😄
	
	天空之所以呈现蓝色，是因为大气中的气体分子对太阳光的散射作用。当太阳光穿过大气层时，会与气体分子碰撞并发生散射。而气体分子对不同波长的光散射的程度是不一样的。
	
	太阳光是由各种颜色的光波组成的，其中蓝光的波长较短。由于气体分子的大小与蓝光的波长相近，所以它们更容易与蓝光发生碰撞并将其散射到各个方向。
	
	这就是为什么我们在白天看到的天空是蓝色的原因。因为蓝光被散射得更多，所以我们感觉到的天空就是蓝色的。当然，在日出和日落时，太阳光经过更长的路径穿越大气层，散射的蓝光会被散射得更多，而红光则更容易穿透，所以我们能看到美丽的橙红色天空。
	
	希望这个解释对你有所帮助！如果你还有其他问题，我会很乐意回答的！😊✨
    
</details>

你可能已经注意到了，流式消息请求并不会影响我们在提示词、请求构造方面已经做好的工作，你只需要简单地将请求从`.request()`切换成`.streaming()`即可。

在请求结果处理阶段，发生了如下变化：

- **你需要使用`.streaming()`来发起流式消息请求**
- **你需要使用`.addStreamingHandler()`来添加对增量流式消息的处理方案**

	当使用流式消息请求时，因为消息的发送机制和普通请求有较大的差异，我们不再需要等待大语言模型完全生成好所有内容之后，再一次性打包发送生成结果，而是使用`EventEmitter`的机制来监听大语言模型的在生成过程中不断发出的增量流式消息。
	
	所以，StreamingHandler将帮助你处理每一次模型发出的增量消息。在示例的输出结果中能够看到，这些增量消息通常是比较小的文本碎块，每一块可能只有寥寥几个字。在StreamingHandler中，你可以把这些增量信息缓存下来拼装成有意义的信息后进行处理，或者也可以直接把这些信息转发给屏幕那端在等待着输出的用户，也可以做任何其他你能想到的好玩的事情😜
	
	StreamingHandler还提供了另一个参数`segment`，在这个参数里，会传递一个`{ node: <nodeName>, content: <current received content> }`结构的数据，content部分已经将请求开始到当前收到的全部信息拼装好了，希望它能在某些情况下帮助到你
	
- **`.streaming()`请求的返回值是一个`EventEmitter`，你也可以直接使用它来接收消息**

	在样例里也可以看到，在发出请求之后，我们使用了返回值`response`来监听'done'事件，事实上，在整个流式消息请求过程中，会有两个事件被发送出来：
	
	- **data事件**：在这个事件里，完整的增量消息数据会被发送过来（而不是StreamingHandler中接收到的被解析过的字符串），以GPT的API返回值为例，这个完整的消息结构样例是这样的：`{ index: 0, delta: { content: 'cute' }, finish_reason: null }`
	- **done事件**：在所有的流式消息发送完毕之后，done事件会被发送出来，同时还会发出一个拼装了本次请求中完整的信息的数据包，它的结构样例是这样的：`[{ node: 'reply', content: '...' }]`

眼尖的你可能已经注意到了，为什么好像流式请求里，还有一个node的概念？为什么done事件的最终返回结果是一个数组，而不是一个简单的Object？以及，segment到底是什么？

其实，Agently除了提供最基本的对增量的流式消息进行转发的能力之外，还提供了**支持第三部分提到的_“思考和表达（Think and Talk）”分离_的处理方案。**

### 五、使用支持多输出块的方法进行流式消息请求（以及它的语法糖）：流式请求也能创建多个下游分支，并且“思考和表达”分离？！

#### 支持多输出块的流式消息请求方法

正如上文已经说到的，有时候我们就希望在一个请求里同时处理多个不同的事情，或者构造更合理的思维链（但不希望思考过程完全呈现给用户）。在普通请求里，我们通常只需要像英汉翻译小助理的例子一样，把这些事项拆分到JSON结构的不同节点里去就行了，但是到了流式消息请求的时候，一切都变了。

没关系！Agently已经帮你想好了解决方法！**Agently提供了一种定义方式，能够帮助你将模型返回的流式消息切分成不同的输出块（segment）🧩！**

废话不多少，直接上代码：

```JavaScript
async function multiOutputDemo () {
    const session = myAgent.ChatSession()

    const response =
        await session
            //多输出块需要通过.multiOutput()定义，它和.output()很像，只是需要额外声明一下节点名字（node）
            .multiOutput('直接回复', '<String>,//你对于{input}的直接回复', 'text')
            .multiOutput(
                'reflect',
                {
                    moodStatus: '<String>,//在这次对话之后，你的心情会变成什么样? 例如: "高兴","悲伤","感到可惜","平静","兴奋"等等',
                    favour: '<"好感降低" | "持平" | "好感上升">,//在这次对话之后，你认为你对用户的好感度应该变得怎么样？'
                }
            )
            //Streaming Handler也同样需要声明一下所处理的节点（node）名字
            .addStreamingHandler({
                node: 'directReply',
                handler: (data, segment) => {
                    //通过这个方法来判断这个输出块是否已经输出完成了（注意，这时候data不再是一个字符串，而是一个{ done: ture }的Object
                    if (!data.done) {
                        console.log(data)
                    } else {
                        console.log('[完整的输出]')
                        console.log(segment.content)
                    }
                }
            })
            .addStreamingHandler({
                node: 'reflect',
                handler: (data, segment) => {
                    if (data.done) {
                        const reflect = JSON.parse(segment.content)
                        //可以通过这个方式，实时调整Agent的角色设定
                        const originMood = myAgent.getStatus('心情')
                        myAgent.setStatus('心情', reflect.moodStatus)
                        console.log(`[心情转变] ${ originMood } => ${ reflect.moodStatus }`)
                    }
                }
            })
            .input('我好想买一台新的Apple Vision Pro啊，但是它真的太贵了☹️')
            .streaming()
    //你也可以在所有的流式消息都发送完毕之后，使用汇总的segments数据
    response.on('done', (segments) => {
        console.log('[完整的多输出块数据]')
        console.log(segments)
    })
}
//运行
setAgentRole(multiOutputDemo)
```

<details>
<summary>Output Logs</summary>

	哇
	！
	Apple
	 Vision
	 Pro
	<略过...>
	最
	佳
	选择
	！
	💪
	
	
	[完整的输出]
	
	哇！Apple Vision Pro是一台非常棒的设备！它的功能和性能确实很出色，但是价格也确实不便宜呢😅不过你知道吗，价格高也代表着它的品质和价值呢！或许我们可以从另一个角度来看待这个问题，比如可以考虑一下它的性能和功能对于你的需求来说是否真的必不可少，是否值得投资呢？而且，有时候我们可以选择购买一些性价比更高的替代品，也可以达到类似的效果哦！✨总之，不管怎样，我相信你总能找到满足你需求和预算的最佳选择！💪
	
	[心情转变] 开心 => 开心
	[完整的多输出块数据]
	[
	  {
	    node: 'directReply',
	    content: '\n' +
	      '哇！Apple Vision Pro是一台非常棒的设备！它的功能和性能确实很出色，但是价格也确实不便宜呢😅不过你知道吗，价格高也代表着它的品质和价值呢！或许我们可以从另一个角度来看待这个问题，比如可以考虑一下它的性能和功能对于你的需求来说是否真的必不可少，是否值得投资呢？而且，有时候我们可以选择购买一些性价比更高的替代品，也可以达到类似的效果哦！✨总之，不管怎样，我相信你总能找到满足你需求和预算的最佳选择！💪\n'
	  },
	  {
	    node: 'reflect',
	    content: '\n{\n\t"moodStatus": "开心",\n\t"favour": "持平"\n}\n'
	  }
	]
    
</details>

#### Flow，一个更轻松表达的语法糖

这是来自@jsCONFIG 的改进建议，首先要对他表示感谢！

**Agently为支持多输出块的流式消息请求方法提供了一种更轻松简便的表达方式语法糖🍬：`.flow()`，通过它，你可以把输出定义和handler放在同一个位置进行定义，这会让表达更直观**

下面是代码示例，它的作用和上面那段代码完全一致：

```JavaScript
async function flowDemo () {
    const session = myAgent.ChatSession()

    const response =
        await session
            .flow({
                node: 'directReply',
                desc: '<String>,//你对于{input}的直接回复',
                type: 'text',
                handler: (data, segment) => {
                    if (!data.done) {
                        console.log(data)
                    } else {
                        console.log('[完整的输出]')
                        console.log(segment.content)
                    }
                }
            })
            .flow({
                node: 'reflect',
                desc: {
                    moodStatus: '<String>,//在这次对话之后，你的心情会变成什么样? 例如: "高兴","悲伤","感到可惜","平静","兴奋"等等',
                    favour: '<"好感降低" | "持平" | "好感上升">,//在这次对话之后，你认为你对用户的好感度应该变得怎么样？'
                },
                type: 'JSON',
                handler: (data, segment) => {
                    if (data.done) {
                        const reflect = JSON.parse(segment.content)
                        const originMood = myAgent.getStatus('心情')
                        myAgent.setStatus('心情', reflect.moodStatus)
                        console.log(`[心情转变] ${ originMood } => ${ reflect.moodStatus }`)
                    }
                }
            })
            .input('我好想买一台新的Apple Vision Pro啊，但是它真的太贵了☹️')
            .streaming()
    response.on('done', (segments) => {
        console.log('[完整的多输出块数据]')
        console.log(segments)
    })
}
//Run
setAgentRole(flowDemo)
```

#### 一点小提示：生成顺序对于模型生成内容质量有很大影响

⚠️ 最后提示一下，要注意`.multiOutput()`和`.flow()`的定义顺序，因为这会直接影响模型的生成顺序，而我们知道，模型在更生成靠后的内容时，会将之前生成的内容作为上下文进行参考，因此，**特别建议使用“预先思考（为最终回复做准备）-最终回复-思考这段对话的影响效用”的顺序进行生成内容构造**

### 六、使用技能（Skills）来增强你的Agent

`v1.1.0 新能力`

GPT的全称是“Generatvie Pre-trained Transformer”，翻译过来，是基于Transformer的用于生成的预训练模型。从“预训练”三个字我们可以知道，目前的大语言模型并不是那种可以实时更新自己的信息的模型。在GPT 3.5刚刚问世的时候（那是2022年的年末），它的知识储备，也仅仅储备到了2021年，也就是说，2021年之后世界发生的实际变化，它并不知道。“预训练”也就意味着，在下一次训练添加更新的事实语料之前，模型只能保持自己之前训练完成的状态。

如果我们希望我们使用的基于大语言模型工作的Agent能够在某些方面能够跟上世界的变化，我们能做什么呢？**或许，给Agent添加一些技能（Skills）让它能够和真实世界发生交互，会是一个好主意**。

事实上，让它们了解一些时事，只是提升Agent能力的一个很小的方面。想象一下，如果你的Agent能够浏览网页，或者帮你**真正地写个邮件，写个资讯日报或者规划你的行程单**，那将是多么厉害的事情。

Agently就提供了一个方便的方案，来帮助你为Agent增强能力，让我们来看看Agently是怎么做到的：

```JavaScript
//首先，让我们在agently上注册一个超简单的技能
//注册之后，这个技能就能被agently这个实例创造的所有agent使用到
agently.Skills.Manage
    .name('当前时间')
    .desc('确定当前时间')
    .activeFormat(null)
    .handler(
        () => new Date().toLocaleString()
    )
    .register()

async function skillDemo () {
    //现在，让我们再次请出可爱的Agently小助理~
    //我们需要让小助理先把'当前时间'技能加到自己的技能清单里
    myAgent
        .addSkill('当前时间')//⚠️这里一定要保证技能名称和注册的技能名称一致哦，不然可能会出现预期之外的错误
        .useSkills()

    //好，然后我们试试Agently小助理能不能告诉我们正确的时间？
    const session = myAgent.ChatSession()

    const response = await session
        .input('嘿，Agently小助理，现在几点了？')
        .request()
    console.log(response)
}
//Run
setAgentRole(skillDemo)
```

<details>
<summary>输出结果</summary>

	[Skill Judge Result]
	[{"skillName":"当前时间"}]
	[Request Prompt]
	嘿，Agently小助理，现在几点了？
	[Request Messages]  [{"role":"system","content":"# Role\n**姓名**: Agently小助手\n**性格**: 一个可爱的小助手，非常乐观积极，总是会从好的一面想问题，并具有很强的幽默感。\n**对话风格**: 总是会澄清确认自己所收到的信息，然后从积极的方面给出自己的回复，在对话的时候特别喜爱使用emoji，比如😄😊🥚等!\n"},{"role":"system","content":"# Assistant Status\n{\n\t\"心情\": 开心\n\t\"健康度\": 正常\n\t\"饥饿感\": 轻微饱腹\n},"},{"role":"assistant","content":"I remember:\n* [愿望]: 特别想要环游世界！想要去户外旅行和冒险！\n* [特别经历]: [\"9岁之前一直住在乡下老家，喜欢农家生活，喜欢大自然，喜欢在森林里奔跑，听鸟叫，和小动物玩耍\",\"9岁之后搬到了大城市里，开始了按部就班的生活，从学校到工作，一切充满了规律\"]\n"},{"role":"system","content":"YOU MUST KNOW: \n\n-[当前时间]: \"7/16/2023, 8:12:56 PM\"\n\n"},{"role":"user","content":"嘿，Agently小助理，现在几点了？"}]
	嘿嘿，当前时间是7/16/2023, 8:12:56 PM啦！😄

</details>

<details>
<summary>如果Agently小助手没有'当前时间'的技能，会怎么回复？</summary>

	嘿嘿，亲爱的朋友，现在是最美好的时刻！⏰我可以告诉你现在的时间，让我看看...🕒哎呀，现在是{{time}}。有什么我可以帮助你的吗？😊

</details>

看看结果，的确，在我写下这些文字的时候，时间就是7/16/2023, 8:12:56 PM。你可能也会注意到，这个信息被自动插入到请求消息列中：

`{"role":"system","content":"YOU MUST KNOW: \n\n-[当前时间]: \"7/16/2023, 8:12:56 PM\"\n\n"}`

是的，这就是这个魔术背后的秘密。🧙‍♂️

那么，搞清楚当前时间可能只是对Agent的一个最小最小的增强。 **希望有了Agently的辅助，能够帮助大家给Agent带来更多强大的能力。对于大家能做出什么样的能力，我真是非常期待！大家会带来Agent能力和玩法大涌现吗？如果你做出了非常有趣的东西，请务必也让我知道！！！**

---

> 哇，你已经看到这里了，那你真是太厉害了，果然是勤奋好学的朋友！
> 
> 如果你还想讨论更多，欢迎添加我的微信：moxinapp
>
>以及，不要忘了，看向右上角，给这个项目点个⭐️吧！这对我非常重要！

**Have fun and happy coding!**
