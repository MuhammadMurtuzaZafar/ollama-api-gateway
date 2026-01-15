import React, { useState } from 'react';
import { Book, Terminal, Code, Copy, CheckCircle, Shield, Key, Zap, BarChart } from 'lucide-react';

const ApiDocs = () => {
  const [copiedCode, setCopiedCode] = useState(null);

  const copyToClipboard = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  const codeExamples = {
    curl: `curl -X POST http://localhost:8000/api/chat \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "smollm-135m",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "top_p": 0.9,
    "num_predict": 2048
  }'`,
    python: `import requests

url = "http://localhost:8000/api/chat"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "model": "smollm-135m",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "top_p": 0.9,
    "num_predict": 2048
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())`,
    javascript: `const response = await fetch('http://localhost:8000/api/chat', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    model: 'smollm-135m',
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Hello!' }
    ],
    temperature: 0.7,
    top_p: 0.9,
    num_predict: 2048
  })
});

const data = await response.json();
console.log(data);`
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center mb-4">
            <Book className="h-10 w-10 text-primary-600 mr-3" />
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
              API Documentation
            </h1>
          </div>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Complete reference for integrating with the EdgeLLM API Gateway
          </p>
        </div>

        {/* Base Endpoint */}
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 mb-8">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
            Base API Endpoint
          </h3>
          <code className="block bg-gray-800 dark:bg-gray-900 text-green-400 p-4 rounded-lg overflow-x-auto">
            http://localhost:8000/api/
          </code>
          <p className="mt-3 text-sm text-gray-600 dark:text-gray-400">
            Include your API key in the <code className="bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded">Authorization: Bearer YOUR_API_KEY</code> header
          </p>
        </div>

        {/* Available Endpoints */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center">
            <Terminal className="h-6 w-6 mr-2 text-primary-600" />
            Available Endpoints
          </h2>
          
          <div className="space-y-6">
            {/* Chat Completion Endpoint */}
            <div className="border-l-4 border-primary-600 pl-4">
              <div className="flex items-center mb-2">
                <span className="bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 px-3 py-1 rounded text-sm font-semibold mr-3">
                  POST
                </span>
                <code className="text-lg font-mono text-gray-900 dark:text-white">/api/chat</code>
              </div>
              <p className="text-gray-600 dark:text-gray-300 mb-3">
                Generate AI chat completions using EdgeLLM with deterministic latency
              </p>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">Request Body:</p>
                <pre className="text-xs text-gray-600 dark:text-gray-400 overflow-x-auto">
{`{
  "model": "smollm-135m",           // Required: Model name
  "messages": [                        // Required: Array of messages
    {
      "role": "system|user|assistant", // Required: Message role
      "content": "string"              // Required: Message content
    }
  ],
  "temperature": 0.7,                  // Optional: 0.0-2.0 (default: 0.7)
  "top_p": 0.9,                        // Optional: 0.0-1.0 (default: 0.9)
  "num_predict": 2048                  // Optional: Max tokens (default: 2048)
}`}
                </pre>
              </div>
            </div>

            {/* List Models Endpoint */}
            <div className="border-l-4 border-blue-600 pl-4">
              <div className="flex items-center mb-2">
                <span className="bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 px-3 py-1 rounded text-sm font-semibold mr-3">
                  GET
                </span>
                <code className="text-lg font-mono text-gray-900 dark:text-white">/api/models</code>
              </div>
              <p className="text-gray-600 dark:text-gray-300 mb-3">
                Get list of available EdgeLLM models
              </p>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">Response:</p>
                <pre className="text-xs text-gray-600 dark:text-gray-400 overflow-x-auto">
{`{
  "models": [
    {
      "name": "smollm-135m",
      "modified_at": "2026-01-12T10:30:00Z",
      "size": 41600000
    },
    {
      "name": "qwen2-0.5b",
      "modified_at": "2026-01-12T10:30:00Z",
      "size": 156000000
    }
  ]
}`}
                </pre>
              </div>
            </div>

            {/* Stats Endpoint */}
            <div className="border-l-4 border-purple-600 pl-4">
              <div className="flex items-center mb-2">
                <span className="bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 px-3 py-1 rounded text-sm font-semibold mr-3">
                  GET
                </span>
                <code className="text-lg font-mono text-gray-900 dark:text-white">/api/stats</code>
              </div>
              <p className="text-gray-600 dark:text-gray-300 mb-3">
                Get usage statistics for your API key
              </p>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <p className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">Response:</p>
                <pre className="text-xs text-gray-600 dark:text-gray-400 overflow-x-auto">
{`{
  "total_requests": 150,
  "total_tokens": 25000,
  "average_response_time": 1.5,
  "last_used": "2025-12-23T12:00:00Z"
}`}
                </pre>
              </div>
            </div>

            {/* API Keys Endpoint (Admin Only) */}
            <div className="border-l-4 border-red-600 pl-4">
              <div className="flex items-center mb-2">
                <span className="bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 px-3 py-1 rounded text-sm font-semibold mr-3">
                  GET
                </span>
                <code className="text-lg font-mono text-gray-900 dark:text-white">/api/keys</code>
                <span className="ml-3 bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 px-2 py-1 rounded text-xs font-semibold">
                  Admin Only
                </span>
              </div>
              <p className="text-gray-600 dark:text-gray-300">
                List all API keys (requires admin role)
              </p>
            </div>
          </div>
        </div>

        {/* Code Examples */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 flex items-center">
            <Code className="h-6 w-6 mr-2 text-primary-600" />
            Code Examples
          </h2>

          {/* cURL Example */}
          <div className="mb-8">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-lg font-semibold text-gray-900 dark:text-white">cURL</h4>
              <button
                onClick={() => copyToClipboard(codeExamples.curl, 'curl')}
                className="flex items-center space-x-2 px-3 py-1.5 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-600 text-sm"
              >
                {copiedCode === 'curl' ? (
                  <>
                    <CheckCircle className="h-4 w-4 text-green-600" />
                    <span>Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
              <code>{codeExamples.curl}</code>
            </pre>
          </div>

          {/* Python Example */}
          <div className="mb-8">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-lg font-semibold text-gray-900 dark:text-white">Python</h4>
              <button
                onClick={() => copyToClipboard(codeExamples.python, 'python')}
                className="flex items-center space-x-2 px-3 py-1.5 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-600 text-sm"
              >
                {copiedCode === 'python' ? (
                  <>
                    <CheckCircle className="h-4 w-4 text-green-600" />
                    <span>Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
              <code>{codeExamples.python}</code>
            </pre>
          </div>

          {/* JavaScript Example */}
          <div className="mb-8">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-lg font-semibold text-gray-900 dark:text-white">JavaScript</h4>
              <button
                onClick={() => copyToClipboard(codeExamples.javascript, 'javascript')}
                className="flex items-center space-x-2 px-3 py-1.5 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-600 text-sm"
              >
                {copiedCode === 'javascript' ? (
                  <>
                    <CheckCircle className="h-4 w-4 text-green-600" />
                    <span>Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
            <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
              <code>{codeExamples.javascript}</code>
            </pre>
          </div>
        </div>

        {/* Authentication & Best Practices */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Authentication & Best Practices
          </h2>
          
          <div className="space-y-4 text-gray-600 dark:text-gray-300">
            <div className="flex items-start">
              <Shield className="h-5 w-5 text-primary-600 mr-3 mt-1 flex-shrink-0" />
              <div>
                <p className="font-semibold text-gray-900 dark:text-white">Use Bearer Token Authentication</p>
                <p className="text-sm">Always include your API key in the Authorization header: <code className="bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded text-xs">Authorization: Bearer YOUR_API_KEY</code></p>
              </div>
            </div>
            
            <div className="flex items-start">
              <Key className="h-5 w-5 text-primary-600 mr-3 mt-1 flex-shrink-0" />
              <div>
                <p className="font-semibold text-gray-900 dark:text-white">Keep Your API Key Secure</p>
                <p className="text-sm">Never commit API keys to version control. Use environment variables or secure vaults.</p>
              </div>
            </div>
            
            <div className="flex items-start">
              <Zap className="h-5 w-5 text-primary-600 mr-3 mt-1 flex-shrink-0" />
              <div>
                <p className="font-semibold text-gray-900 dark:text-white">Rate Limiting</p>
                <p className="text-sm">Implement exponential backoff for retries. Check response headers for rate limit information.</p>
              </div>
            </div>
            
            <div className="flex items-start">
              <BarChart className="h-5 w-5 text-primary-600 mr-3 mt-1 flex-shrink-0" />
              <div>
                <p className="font-semibold text-gray-900 dark:text-white">Monitor Usage</p>
                <p className="text-sm">Regularly check your dashboard for usage statistics and optimize your requests.</p>
              </div>
            </div>
          </div>
        </div>

        {/* Error Responses */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Error Responses
          </h2>
          
          <div className="space-y-4">
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
              <p className="font-semibold text-red-900 dark:text-red-200 mb-2">401 Unauthorized</p>
              <pre className="text-xs text-red-800 dark:text-red-300 overflow-x-auto">
{`{
  "detail": "Invalid API key"
}`}
              </pre>
            </div>
            
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
              <p className="font-semibold text-yellow-900 dark:text-yellow-200 mb-2">429 Too Many Requests</p>
              <pre className="text-xs text-yellow-800 dark:text-yellow-300 overflow-x-auto">
{`{
  "detail": "Rate limit exceeded"
}`}
              </pre>
            </div>
            
            <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-4">
              <p className="font-semibold text-orange-900 dark:text-orange-200 mb-2">400 Bad Request</p>
              <pre className="text-xs text-orange-800 dark:text-orange-300 overflow-x-auto">
{`{
  "detail": "Invalid request parameters"
}`}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ApiDocs;
