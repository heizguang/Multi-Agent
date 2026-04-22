// ===== 全局状态管理 =====
const AppState = {
    userId: null,
    sessionId: null,
    isLoading: false,
    messageHistory: [],
    userInfo: null,
    chartInstances: {}  // ECharts 实例缓存，用于响应式调整
};

// ===== API 配置 =====
const API_BASE_URL = window.location.origin;

// ===== API 调用函数 =====
async function apiCall(endpoint, data = {}) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || '请求失败');
        return result;
    } catch (error) {
        console.error('API调用失败:', error);
        throw error;
    }
}

// ===== 登录相关 =====
async function handleLogin() {
    const userIdInput = document.getElementById('userIdInput');
    const userId = userIdInput.value.trim() || 'guest';
    
    const loginBtn = document.getElementById('loginBtn');
    loginBtn.disabled = true;
    loginBtn.textContent = '登录中...';
    
    try {
        const result = await apiCall('login', { user_id: userId });
        
        if (result.success) {
            AppState.userId = result.user_id;
            AppState.sessionId = result.session_id;
            AppState.userInfo = result.user_info || null;
            
            document.getElementById('loginOverlay').style.display = 'none';
            document.getElementById('mainApp').style.display = 'flex';
            
            updateUserInfo();
            addSystemMessage(`欢迎回来，${userId}！我已准备好为您服务。`);

            if (AppState.userInfo) {
                const prefCount = AppState.userInfo.preferences ? Object.keys(AppState.userInfo.preferences).length : 0;
                const knowCount = Array.isArray(AppState.userInfo.knowledge) ? AppState.userInfo.knowledge.length : 0;
                if (prefCount > 0 || knowCount > 0) {
                    addSystemMessage(`已加载您的长期记忆：偏好 ${prefCount} 项，知识 ${knowCount} 条。`);
                }
            }
        }
    } catch (error) {
        alert('登录失败: ' + error.message);
        loginBtn.disabled = false;
        loginBtn.innerHTML = `
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"/>
                <polyline points="10 17 15 12 10 7"/>
                <line x1="15" y1="12" x2="3" y2="12"/>
            </svg>
            开始使用
        `;
    }
}

function updateUserInfo() {
    document.getElementById('userName').textContent = AppState.userId || 'Guest';
    document.getElementById('sessionId').textContent = 
        AppState.sessionId ? AppState.sessionId.substring(0, 8) + '...' : '-';
}

function handleLogout() {
    if (confirm('确定要退出登录吗？')) {
        location.reload();
    }
}

// ===== Markdown 渲染 =====
function renderMarkdown(mdText) {
    try {
        if (typeof marked !== 'undefined') {
            if (!window.__markedConfigured) {
                marked.setOptions({
                    gfm: true,
                    breaks: true,
                    mangle: false,
                    headerIds: false,
                    highlight: function(code, lang) {
                        try {
                            if (typeof hljs !== 'undefined') {
                                if (lang && hljs.getLanguage(lang)) {
                                    return hljs.highlight(code, { language: lang }).value;
                                }
                                return hljs.highlightAuto(code).value;
                            }
                        } catch (e) {}
                        return code;
                    }
                });
                window.__markedConfigured = true;
            }
            let html = marked.parse(mdText);
            if (typeof DOMPurify !== 'undefined') {
                html = DOMPurify.sanitize(html);
            }
            return html;
        }
    } catch (e) {
        console.warn('Markdown 渲染失败:', e);
    }
    return escapeHtml(mdText);
}

// ===== ECharts 图表渲染 =====
function renderChart(container, chartConfig) {
    if (!container || !chartConfig || typeof echarts === 'undefined') return;
    
    try {
        const chartId = container.id;
        
        // 销毁旧实例
        if (AppState.chartInstances[chartId]) {
            AppState.chartInstances[chartId].dispose();
        }
        
        container.style.height = '320px';
        container.style.width = '100%';
        
        const chart = echarts.init(container, null, { renderer: 'canvas' });
        
        // 注入通用样式
        const defaultConfig = {
            backgroundColor: 'transparent',
            grid: { left: '3%', right: '4%', bottom: '8%', containLabel: true },
            tooltip: { trigger: 'axis' },
            ...chartConfig
        };
        
        chart.setOption(defaultConfig);
        AppState.chartInstances[chartId] = chart;
        
        // 响应式
        const resizeObserver = new ResizeObserver(() => chart.resize());
        resizeObserver.observe(container);
        
    } catch (e) {
        console.warn('图表渲染失败:', e);
        container.innerHTML = `<div style="padding:8px;color:#888;font-size:12px;">图表渲染失败: ${e.message}</div>`;
    }
}

// ===== 消息相关 =====
function addMessage(text, isUser = false, meta = {}) {
    const chatMessages = document.getElementById('chatMessages');
    
    const welcomeMessage = chatMessages.querySelector('.welcome-message');
    if (welcomeMessage) welcomeMessage.remove();
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    
    const now = new Date();
    const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
    
    const bubbleInner = isUser 
        ? `${escapeHtml(text)}`
        : `<div class="markdown-body">${renderMarkdown(text)}</div>`;

    messageDiv.innerHTML = `
        <div class="message-avatar">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                ${isUser 
                    ? '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>'
                    : '<path d="M12 2L2 7L12 12L22 7L12 2Z"/><path d="M2 17L12 22L22 17"/><path d="M2 12L12 17L22 12"/>'
                }
            </svg>
        </div>
        <div class="message-content">
            <div class="message-bubble">${bubbleInner}</div>
            <div class="message-time">${timeStr}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    if (!isUser && typeof hljs !== 'undefined') {
        messageDiv.querySelectorAll('pre code').forEach(block => {
            try { hljs.highlightElement(block); } catch (e) {}
        });
    }
    
    AppState.messageHistory.push({ text, isUser, time: timeStr });
    return messageDiv;
}

function addSystemMessage(text) {
    addMessage('ℹ️ ' + text, false);
}

// ===== 流式消息构建器 =====
function createStreamingMessage() {
    const chatMessages = document.getElementById('chatMessages');
    
    const welcomeMessage = chatMessages.querySelector('.welcome-message');
    if (welcomeMessage) welcomeMessage.remove();
    
    const msgId = 'stream_' + Date.now();
    const now = new Date();
    const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = msgId;
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2L2 7L12 12L22 7L12 2Z"/>
                <path d="M2 17L12 22L22 17"/>
                <path d="M2 12L12 17L22 12"/>
            </svg>
        </div>
        <div class="message-content">
            <div class="message-bubble">
                <div class="stream-status" id="${msgId}_status">
                    <div class="loading-indicator">
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                    </div>
                    <span class="status-text">正在处理...</span>
                </div>
                <div class="stream-sql-container" id="${msgId}_sql" style="display:none;"></div>
                <div class="stream-sources" id="${msgId}_sources" style="display:none;"></div>
                <div class="markdown-body stream-answer" id="${msgId}_answer"></div>
                <div class="stream-chart" id="${msgId}_chart" style="display:none;"></div>
                <div class="stream-duration" id="${msgId}_duration" style="display:none;"></div>
            </div>
            <div class="message-time">${timeStr}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return {
        msgId,
        messageDiv,
        
        setStatus(text) {
            const el = document.getElementById(`${msgId}_status`);
            if (el) {
                el.innerHTML = `
                    <div class="loading-indicator">
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                    </div>
                    <span class="status-text">${escapeHtml(text)}</span>
                `;
            }
            chatMessages.scrollTop = chatMessages.scrollHeight;
        },
        
        hideStatus() {
            const el = document.getElementById(`${msgId}_status`);
            if (el) el.style.display = 'none';
        },
        
        showSQL(sql, retryCount) {
            const el = document.getElementById(`${msgId}_sql`);
            if (!el) return;
            const retryBadge = retryCount > 0 
                ? `<span class="retry-badge">自动修复 ${retryCount} 次</span>` 
                : '';
            el.style.display = 'block';
            el.innerHTML = `
                <details class="sql-details">
                    <summary>查看生成的 SQL ${retryBadge}</summary>
                    <pre><code class="language-sql">${escapeHtml(sql)}</code></pre>
                </details>
            `;
            if (typeof hljs !== 'undefined') {
                el.querySelectorAll('pre code').forEach(b => {
                    try { hljs.highlightElement(b); } catch(e) {}
                });
            }
        },
        
        showSources(sources) {
            const el = document.getElementById(`${msgId}_sources`);
            if (!el || !sources || !sources.length) return;
            const validSources = sources.filter(s => s && s.length > 0).slice(0, 5);
            if (!validSources.length) return;
            el.style.display = 'block';
            el.innerHTML = `
                <div class="sources-header">🌐 参考来源</div>
                <ul class="sources-list">
                    ${validSources.map((url, i) => {
                        const domain = (() => { try { return new URL(url).hostname; } catch { return url; } })();
                        return `<li><a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">[${i+1}] ${escapeHtml(domain)}</a></li>`;
                    }).join('')}
                </ul>
            `;
        },
        
        appendChunk(chunk) {
            const el = document.getElementById(`${msgId}_answer`);
            if (!el) return;
            // 累积原始文本，然后重新渲染 Markdown
            el._rawText = (el._rawText || '') + chunk;
            el.innerHTML = renderMarkdown(el._rawText);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        },
        
        showChart(chartConfig) {
            const el = document.getElementById(`${msgId}_chart`);
            if (!el || !chartConfig) return;
            el.style.display = 'block';
            el.id = `${msgId}_chart_canvas`;
            renderChart(el, chartConfig);
        },

        showDuration(durationMs) {
            const el = document.getElementById(`${msgId}_duration`);
            if (!el) return;

            const ms = Math.max(0, Number(durationMs) || 0);
            const seconds = ms / 1000;
            let durationText = `${seconds.toFixed(2)} 秒`;

            if (seconds >= 60) {
                const totalSec = Math.round(seconds);
                const minutes = Math.floor(totalSec / 60);
                const remain = totalSec % 60;
                durationText = `${minutes} 分 ${remain} 秒`;
            }

            el.style.display = 'block';
            el.innerHTML = `
                <span class="duration-label">总时长</span>
                <span class="duration-value">${durationText}</span>
            `;
        },
        
        finalize(fullAnswer) {
            this.hideStatus();
            AppState.messageHistory.push({ text: fullAnswer, isUser: false, time: timeStr });
        }
    };
}

function showLoading() {
    const chatMessages = document.getElementById('chatMessages');
    
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant';
    loadingDiv.id = 'loadingMessage';
    
    loadingDiv.innerHTML = `
        <div class="message-avatar">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2L2 7L12 12L22 7L12 2Z"/>
                <path d="M2 17L12 22L22 17"/>
                <path d="M2 12L12 17L22 12"/>
            </svg>
        </div>
        <div class="message-content">
            <div class="message-bubble">
                <div class="loading-indicator">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideLoading() {
    const loadingMessage = document.getElementById('loadingMessage');
    if (loadingMessage) loadingMessage.remove();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, '<br>');
}

// ===== 查询处理（流式版本）=====
async function handleQuery(question) {
    if (!question.trim()) return;
    
    if (AppState.isLoading) {
        alert('请等待当前查询完成');
        return;
    }
    
    AppState.isLoading = true;
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = true;
    
    // 显示用户消息
    addMessage(question, true);
    
    // 创建流式消息容器
    const streamMsg = createStreamingMessage();
    const queryStartAt = performance.now();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/query_stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: AppState.userId,
                question: question
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        let fullAnswer = '';
        let pendingChart = null;
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            // 按 SSE 分隔符拆分事件
            const parts = buffer.split('\n\n');
            buffer = parts.pop(); // 保留不完整的最后一段
            
            for (const part of parts) {
                const line = part.trim();
                if (!line.startsWith('data: ')) continue;
                
                try {
                    const event = JSON.parse(line.slice(6));
                    
                    switch (event.type) {
                        case 'status':
                            streamMsg.setStatus(event.message || '处理中...');
                            break;
                        
                        case 'intent':
                            // 根据意图调整状态提示
                            const intentLabels = {
                                'simple_answer': '简单回答',
                                'sql_only': 'SQL 查询',
                                'analysis_only': '数据分析',
                                'sql_and_analysis': 'SQL + 分析',
                                'web_search': '🌐 联网搜索',
                                'search_and_sql': '🌐 搜索 + SQL 对比'
                            };
                            const label = intentLabels[event.intent] || event.intent;
                            streamMsg.setStatus(`意图识别：${label}`);
                            break;
                        
                        case 'sql':
                            streamMsg.showSQL(event.sql || '', event.retry_count || 0);
                            break;
                        
                        case 'sources':
                            streamMsg.showSources(event.sources || []);
                            break;
                        
                        case 'chart':
                            // 先缓存图表配置，等文字回答结束后再渲染（避免布局抖动）
                            pendingChart = event.config;
                            break;
                        
                        case 'chunk':
                            streamMsg.hideStatus();
                            streamMsg.appendChunk(event.content || '');
                            fullAnswer += (event.content || '');
                            break;
                        
                        case 'error':
                            console.warn('SSE error event:', event.message);
                            streamMsg.setStatus('⚠️ ' + (event.message || '发生错误'));
                            break;
                        
                        case 'done':
                            if (event.answer && !fullAnswer) {
                                fullAnswer = event.answer;
                                streamMsg.appendChunk(fullAnswer);
                            }
                            // 渲染图表（在答案之后）
                            if (pendingChart) {
                                streamMsg.showChart(pendingChart);
                            }
                            streamMsg.showDuration(performance.now() - queryStartAt);
                            streamMsg.finalize(fullAnswer);
                            break;
                    }
                } catch (parseErr) {
                    console.warn('SSE 事件解析失败:', parseErr, line);
                }
            }
        }
        
        // 渲染代码高亮
        if (typeof hljs !== 'undefined') {
            document.getElementById(streamMsg.msgId)?.querySelectorAll('pre code').forEach(b => {
                try { hljs.highlightElement(b); } catch(e) {}
            });
        }
        
    } catch (error) {
        streamMsg.hideStatus();
        streamMsg.appendChunk('抱歉，发生错误：' + error.message);
        streamMsg.finalize('');
        console.error('流式查询失败:', error);
    } finally {
        AppState.isLoading = false;
        sendBtn.disabled = false;
    }
}

function handleSend() {
    const input = document.getElementById('questionInput');
    const question = input.value.trim();
    
    if (question) {
        handleQuery(question);
        input.value = '';
        input.style.height = 'auto';
    }
}

// ===== 会话管理 =====
async function handleNewSession() {
    if (!confirm('确定要开始新会话吗？当前对话历史将被清空（您的长期记忆会保留）。')) return;
    
    try {
        const result = await apiCall('new_session', { user_id: AppState.userId });
        
        if (result.success) {
            AppState.sessionId = result.session_id;
            AppState.messageHistory = [];
            
            // 销毁所有图表实例
            Object.values(AppState.chartInstances).forEach(chart => {
                try { chart.dispose(); } catch (e) {}
            });
            AppState.chartInstances = {};
            
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.innerHTML = `
                <div class="welcome-message">
                    <h2>🔄 新会话已开始</h2>
                    <p>您可以开始新的对话了。</p>
                </div>
            `;
            
            updateUserInfo();
            addSystemMessage('新会话已创建，会话ID: ' + result.session_id.substring(0, 8) + '...');
        }
    } catch (error) {
        alert('创建新会话失败: ' + error.message);
    }
}

// ===== 用户信息 =====
async function handleShowUserInfo() {
    try {
        let userInfo = AppState.userInfo;
        if (!userInfo) {
            const result = await apiCall('user_info', { user_id: AppState.userId });
            if (result.success) {
                userInfo = result.user_info;
                AppState.userInfo = userInfo;
            }
        }
        
        if (userInfo) {
            const modal = document.getElementById('userInfoModal');
            const content = document.getElementById('userInfoContent');
            
            let html = `
                <div class="info-item">
                    <div class="info-label">用户ID</div>
                    <div class="info-value">${userInfo.user_id || '-'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">会话ID</div>
                    <div class="info-value">${userInfo.session_id || '-'}</div>
                </div>
            `;

            if (userInfo.profile) {
                html += `
                    <div class="info-item">
                        <div class="info-label">创建时间</div>
                        <div class="info-value">${userInfo.profile.created_at || '-'}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">最后活跃</div>
                        <div class="info-value">${userInfo.profile.last_active || '-'}</div>
                    </div>
                `;
            }
            
            if (userInfo.preferences && Object.keys(userInfo.preferences).length > 0) {
                html += `<div class="info-item"><div class="info-label">用户偏好</div><ul class="preferences-list">`;
                for (const [key, value] of Object.entries(userInfo.preferences)) {
                    html += `<li><strong>${key}:</strong> ${value}</li>`;
                }
                html += `</ul></div>`;
            } else {
                html += `
                    <div class="info-item">
                        <div class="info-label">用户偏好</div>
                        <div class="info-value" style="color: var(--text-tertiary);">
                            暂无偏好记录。继续使用系统，我们会自动学习您的偏好。
                        </div>
                    </div>
                `;
            }

            const knowledge = Array.isArray(userInfo.knowledge) ? userInfo.knowledge : [];
            if (knowledge.length > 0) {
                html += `<div class="info-item"><div class="info-label">用户知识（最近${knowledge.length}条）</div><ul class="preferences-list">`;
                knowledge.slice(0, 20).forEach(k => {
                    const summary = (k.content || '').length > 120 ? k.content.slice(0, 120) + '...' : (k.content || '');
                    html += `<li><strong>${k.category || '知识'}:</strong> ${summary}</li>`;
                });
                html += `</ul></div>`;
            }
            
            content.innerHTML = html;
            modal.classList.add('active');
        }
    } catch (error) {
        alert('获取用户信息失败: ' + error.message);
    }
}

function handleCloseModal() {
    document.getElementById('userInfoModal').classList.remove('active');
}

// ===== 快捷问题 =====
function handleQuickQuestion(question) {
    const input = document.getElementById('questionInput');
    input.value = question;
    input.focus();
}

// ===== 事件监听器 =====
document.addEventListener('DOMContentLoaded', () => {
    const userIdInput = document.getElementById('userIdInput');
    const loginBtn = document.getElementById('loginBtn');
    
    userIdInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleLogin(); });
    loginBtn.addEventListener('click', handleLogin);
    
    document.getElementById('sendBtn').addEventListener('click', handleSend);
    document.getElementById('newSessionBtn').addEventListener('click', handleNewSession);
    document.getElementById('userInfoBtn').addEventListener('click', handleShowUserInfo);
    document.getElementById('logoutBtn').addEventListener('click', handleLogout);
    document.getElementById('closeModalBtn').addEventListener('click', handleCloseModal);
    
    const questionInput = document.getElementById('questionInput');
    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });
    questionInput.addEventListener('input', () => {
        questionInput.style.height = 'auto';
        questionInput.style.height = questionInput.scrollHeight + 'px';
    });
    
    document.querySelectorAll('.question-item').forEach(btn => {
        btn.addEventListener('click', () => handleQuickQuestion(btn.getAttribute('data-question')));
    });
    
    const modal = document.getElementById('userInfoModal');
    modal.addEventListener('click', (e) => { if (e.target === modal) handleCloseModal(); });
    
    userIdInput.focus();
    
    // 窗口大小变化时调整图表
    window.addEventListener('resize', () => {
        Object.values(AppState.chartInstances).forEach(chart => {
            try { chart.resize(); } catch (e) {}
        });
    });
});

// ===== 工具函数 =====
function formatTimestamp(date) {
    return `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
}

// ===== 导出API供控制台调试 =====
window.AppDebug = {
    state: AppState,
    apiCall,
    handleQuery,
    handleNewSession
};

console.log('🚀 多智能体数据查询系统前端 v3.0 已加载');
console.log('✨ 新特性: SSE流式响应 | DeepSearch联网搜索 | ECharts可视化 | SQL自动纠错');
console.log('💡 提示：可以通过 window.AppDebug 访问调试API');
