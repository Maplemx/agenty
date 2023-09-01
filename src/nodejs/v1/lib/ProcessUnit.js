const EventEmitter = require('events')

class PrefixManage {
    constructor (ProcessUnit) {
        this.prefix = ProcessUnit.prefix
        this.debug = ProcessUnit.debug
        this.runtime = {}
    }

    generateRole (generateRoleHandler) {
        this.runtime.generateRole = generateRoleHandler
        return this
    }

    generateStatus (generateStatusHandler) {
        this.runtime.generateStatus = generateStatusHandler
        return this
    }

    generateMemories (generateMemoriesHandler) {
        this.runtime.generateMemories = generateMemoriesHandler
        return this
    }    

    generateSkillTips (generateSkillsTipsHandler) {
        this.runtime.generateSkillTips = generateSkillsTipsHandler
        return this
    }

    generateSkillJudgePrompt (generateSkillJudgePromptHandler) {
        this.runtime.generateSkillJudgePrompt = generateSkillJudgePromptHandler
        return this
    }

    generateContext (generateContextHandler) {
        this.runtime.generateContext = generateContextHandler
        return this
    }

    shortenContext (shortenContextHandler) {
        this.runtime.shortenContext = shortenContextHandler
        return this
    }

    generatePrompt (generatePromptHandler) {
        this.runtime.generatePrompt = generatePromptHandler
        return this
    }

    register () {
        let missing = []
        if (!this.runtime.generateRole) missing.push('generateRole')
        if (!this.runtime.generateContext) missing.push('generateContext')
        if (!this.runtime.shortenContext) missing.push('shortenContext')
        if (!this.runtime.generatePrompt) missing.push('generatePrompt')
        if (missing.length === 0) {
            this.prefix.default = { ...this.runtime }
            this.runtime = {}
            return { status: 200 }
        } else {
            const msg = `[Prefix Register] Failed: missing ${ JSON.stringify(missing) }`
            this.debug(msg)
            return { status: 400, msg: msg }
        }
    }

    update () {
        for (let key in this.runtime) {
            this.prefix.default[key] = this.runtime[key]
        }
        this.runtime = {}
        return { status: 200 }
    }
}

class ProcessRequest {
    constructor (AgentSession, LLM, Skills, ProcessUnit) {
        this.getSkillList = Skills.getSkillList
        this.sessionId = AgentSession.id
        this.getAgentSettings = AgentSession.getAgentSettings
        this.getSessionSettings = AgentSession.getSessionSettings
        this.memories = {
            memories: AgentSession.getMemories(),
            getMemory: AgentSession.getMemory,
        }
        this.status = {
            status: AgentSession.getStatus(),
            getStatus: AgentSession.getStatus,
        }
        this.context = AgentSession.context
        this.appendContext = AgentSession.appendContext
        this.getPrompt = AgentSession.getPrompt
        this.runtime = AgentSession.runtime
        this.prefix = ProcessUnit.prefix.default
        //this.responseHandlers = AgentSession.responseHandlers
        this.getResponseHandlers = AgentSession.getResponseHandlers
        //this.streamingHandlers = AgentSession.streamingHandlers
        this.getStreamingHandlers = AgentSession.getStreamingHandlers
        this.initRuntime = AgentSession.initRuntime
        this.debug = AgentSession.debug
        this.llmRequest = LLM.Request(this.getAgentSettings().llmName)
    }

    async start(requestType) {
        let requestMessages = [],
            skillJudgeMessages = []
        //Generate Role Messages
        if (Object.keys(this.getAgentSettings().role).length > 0) {
            requestMessages = requestMessages.concat(this.prefix.generateRole(this.getAgentSettings().role))
        }
        //Generate Status Messages
        if (this.getAgentSettings().useStatus && this.prefix.generateStatus) {
            requestMessages = requestMessages.concat(this.prefix.generateStatus(this.status))
        }
        //Generate Memories Messages
        if (this.getAgentSettings().useMemory && this.prefix.generateMemories) {
            requestMessages = requestMessages.concat(this.prefix.generateMemories(this.memories))
        }
        //Generate Skills Messages
        if (this.getAgentSettings().useSkills && this.prefix.generateSkillTips && this.getAgentSettings().skills.length > 0) {
            skillJudgeMessages = skillJudgeMessages.concat(this.prefix.generateSkillTips(this.getAgentSettings().skills, this.getSkillList()))
        }
        if (this.getSessionSettings().loadContext) {
            //Generate Context Messages
            let contextMessages = this.prefix.generateContext(this.context)
            //Shorten Context
            if (JSON.stringify(contextMessages).length > this.getSessionSettings().maxContextLength) {
                contextMessages = await this.prefix.shortenContext(contextMessages, this.context.full, this.getSessionSettings().maxContextLength)
            }
            this.context.request = contextMessages
            requestMessages = requestMessages.concat(contextMessages)
            skillJudgeMessages = skillJudgeMessages.concat(contextMessages)
        }
        //Generate Skill Judge Prompt
        if (this.getAgentSettings().useSkills && this.prefix.generateSkillJudgePrompt && Object.keys(this.getSkillList()).length > 0 && this.getAgentSettings().skills.length > 0) {
            const skillPromptMessage = this.prefix.generateSkillJudgePrompt(this.getPrompt(), this.getSkillList())
            skillJudgeMessages = skillJudgeMessages.concat(skillPromptMessage)
            this.debug(`[Skill Judge Prompt]`)
            this.debug(skillPromptMessage.content)
            let judgeResponse = await this.llmRequest['request'](skillJudgeMessages, this.getAgentSettings().requestOptions || {})
            this.debug(`[Skill Judge Result]`)
            this.debug(judgeResponse)
            judgeResponse = judgeResponse
                .replace(/```.+\n/ig, '')
                .replaceAll('```', '')
                .trim()
            try {
                let parsedJudgeResponse = JSON.parse(judgeResponse)
                if(!Array.isArray(parsedJudgeResponse)) {
                    parsedJudgeResponse = [].concat(parsedJudgeResponse)
                }
                if (parsedJudgeResponse.length > 0) {
                    const skillList = this.getSkillList()
                    let skillResponseMessage = { role: 'system', content: 'YOU MUST KNOW: \n\n' }
                    for (let i = 0; i < parsedJudgeResponse.length; i++) {
                        skillResponseMessage.content += `-[${ parsedJudgeResponse[i].skillName }]: ${ JSON.stringify(await skillList[parsedJudgeResponse[i].skillName].handler(parsedJudgeResponse[i].data)) }\n\n`
                    }
                    requestMessages = requestMessages.concat(skillResponseMessage)
                }
            } catch (e) {
                this.debug(`[Skill Judge Response Parse Failed] ${ e.message }`)
            }
        }

        //Generate Request Prompt
        const promptMessage = this.prefix.generatePrompt(this.getPrompt())
        this.debug(`[Request Prompt]`)
        this.debug(promptMessage.content)
        requestMessages = requestMessages.concat(promptMessage)
        //Request LLM
        let response = await this.llmRequest[requestType](requestMessages, this.getAgentSettings().requestOptions || {})
        //Handle Response
        //<direct request>
        if (requestType === 'request') {
            //Clean Response
            response = response
                .replace(/```.+\n/ig, '')
                .replace('```', '')
            //Call Handlers
            for (let i = 0; i < this.getResponseHandlers().length; i++) {
                await this.getResponseHandlers()[i](response, (content) => { this.runtime.reply = content })
            }
            //Save Context
            const finalResponse = this.runtime.reply || response
            if (this.getSessionSettings().saveContext && finalResponse) {
                this.appendContext([
                    this.getSessionSettings().saveFullPrompt ? promptMessage : { role: 'user', content: typeof(this.getPrompt().input) === 'string' ? this.getPrompt().input : JSON.stringify(this.getPrompt().input) },//If session.setSaveFullPrompt(true) is state, the full prompt will be saved, otherwise only input part will be saved.
                    { role: 'assistant', content: finalResponse }
                ])
            }
            //Init Runtime
                this.initRuntime()
            //Final Response
            return finalResponse
        //<streaming>
        } else if (requestType === 'streaming') {
            //Create response event listerners
            const responseEvent = new EventEmitter()
            let replyNode
            for (let i = 0; i < this.getStreamingHandlers().length; i++) {
                responseEvent.on(this.getStreamingHandlers()[i].node || 'data', this.getStreamingHandlers()[i].handler || this.getStreamingHandlers()[i])
                if (this.getStreamingHandlers()[i].reply) replyNode = this.getStreamingHandlers()[i].node
            }
            let buffer = '',
                isInSegment = false,
                currentNode = '',
                segment = { node: null, content: '' },
                segments = []
            if (this.getPrompt().multiOutput.length === 0) {
                segment.node = 'reply'
                replyNode = 'reply'
            }
            response.on('data', (data) => {
                const delta = data.delta.content
                if (typeof(delta) === 'string') {
                    //Single Output
                    if (this.getPrompt().multiOutput.length === 0) {
                        segment.content += delta
                        responseEvent.emit('data', delta, segment)
                    //Multi Output
                    } else {
                        buffer += delta
                        if (!isInSegment) {
                            const startMarkSearcher = buffer.match(/<\$\$\$node=.+>/ig)
                            if (startMarkSearcher && startMarkSearcher.length > 0){
                                currentNode = startMarkSearcher[0].substr(9)
                                currentNode = currentNode.substr(0, currentNode.length - 1)
                                buffer = buffer.replace(/[\s\S]*<\$\$\$node=.+>/ig, '')
                                segment.node = currentNode
                                segment.content = buffer
                                responseEvent.emit(currentNode, buffer, segment)
                                buffer = ''
                                isInSegment = true
                            }
                        } else {
                            if (Math.sign(buffer.indexOf('<')) + Math.sign(buffer.indexOf('<\/')) + Math.sign(buffer.indexOf('<\/$')) + Math.sign(buffer.indexOf('<\/$$')) + Math.sign(buffer.indexOf('<\/$$$')) + Math.sign(buffer.indexOf('<\/$$$>')) > -6) {
                                const content = buffer.substr(0, buffer.indexOf('<'))
                                if (content.length > 0) {
                                    segment.content += content
                                    responseEvent.emit(currentNode, content, segment)
                                }
                                buffer = buffer.substr(buffer.indexOf('<'), buffer.length)
                                if (buffer.length >= 6 && buffer.substr(0, 6) !== '<\/$$$>') {
                                    segment.content += buffer
                                    responseEvent.emit(currentNode, buffer, segment)
                                    buffer = ''
                                } else {
                                    buffer = buffer.replace('<\/$$$>')
                                    //console.log(`[${currentNode}]: done`)
                                    responseEvent.emit(currentNode, { done: true }, segment)
                                    segments.push({ ...segment })
                                    segment = { node: null, content: '' }
                                    isInSegment = false
                                }
                            } else {
                                segment.content += delta
                                responseEvent.emit(currentNode, delta, segment)
                                buffer = ''
                            }
                        }
                    }
                }
            })
            response.on('finish', (completeReply) => {
                if (segment.node && segment.content !== '') segments.push({ ...segment })
                //Save Context
                if (this.getSessionSettings().saveContext) {
                    let content
                    if (this.getPrompt().multiOutput.length === 0) {
                        content = segment.content
                    } else {                        
                        for (let i = 0; i < segments.length; i++) {
                            if (segments[i].node === replyNode) content = segments[i].content
                        }
                        content = content || JSON.stringify(segments)
                    }
                    this.appendContext([
                        this.getSessionSettings().saveFullPrompt ? promptMessage : { role: 'user', content: typeof(this.getPrompt().input) === 'string' ? this.getPrompt().input : JSON.stringify(this.getPrompt().input) },//If session.setSaveFullPrompt(true) is state, the full prompt will be saved, otherwise only input part will be saved.
                        { role: 'assistant', content: content }
                    ])
                }
                //Init Runtime
                this.initRuntime()
                //Emit Done
                response.emit('done', segments)
            })
            return response
        }
    }
}

class ProcessUnit {
    constructor (Agently) {
        this.Options = Agently.Options
        this.debug = Agently.debug
        this.prefix = {}
        this.Manage = new PrefixManage(this)
        this.Request = (AgentSession) => new ProcessRequest(AgentSession, Agently.LLM, Agently.Skills, this)
    }
}

module.exports = ProcessUnit