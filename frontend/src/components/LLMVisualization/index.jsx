import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, RotateCcw, SkipForward, ChevronRight, Zap, Cpu, Database, Layers, HardDrive, Monitor, ArrowRight } from 'lucide-react';

// Main visualization component
export default function LLMVisualization() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentToken, setCurrentToken] = useState(0);
  // Separate layer tracking for each system
  const [ollamaLayer, setOllamaLayer] = useState(0);
  const [edgeLayer, setEdgeLayer] = useState(0);
  const [ollamaStep, setOllamaStep] = useState(0);
  const [edgeStep, setEdgeStep] = useState(0);
  const [ollamaOutput, setOllamaOutput] = useState([]);
  const [edgeOutput, setEdgeOutput] = useState([]);
  const [ollamaTime, setOllamaTime] = useState(0);
  const [edgeTime, setEdgeTime] = useState(0);
  const [speed, setSpeed] = useState(1);
  
  const inputTokens = ['The', 'cat', 'sat', 'on'];
  const expectedOutput = ['the', 'mat', '.'];
  const totalLayers = 6;
  const stepsPerLayer = 7; // GEMV operations per layer
  
  // EdgeLLM animation - runs ~2.5x faster
  useEffect(() => {
    if (!isPlaying) return;
    
    const edgeInterval = setInterval(() => {
      setEdgeTime(prev => prev + 12.5);
      
      setEdgeStep((prev) => {
        if (prev >= stepsPerLayer - 1) {
          setEdgeLayer((prevLayer) => {
            if (prevLayer >= totalLayers - 1) {
              // Generate token
              setEdgeOutput((prevOut) => {
                const nextToken = expectedOutput[prevOut.length];
                if (nextToken) return [...prevOut, nextToken];
                if (prevOut.length >= expectedOutput.length) setIsPlaying(false);
                return prevOut;
              });
              return 0;
            }
            return prevLayer + 1;
          });
          return 0;
        }
        return prev + 1;
      });
    }, 400 / speed); // Slower interval for EdgeLLM (visible animation)
    
    return () => clearInterval(edgeInterval);
  }, [isPlaying, speed]);
  
  // Ollama animation - runs slower
  useEffect(() => {
    if (!isPlaying) return;
    
    const ollamaInterval = setInterval(() => {
      setOllamaTime(prev => prev + 31.25);
      
      setOllamaStep((prev) => {
        if (prev >= stepsPerLayer - 1) {
          setOllamaLayer((prevLayer) => {
            if (prevLayer >= totalLayers - 1) {
              // Generate token
              setOllamaOutput((prevOut) => {
                const nextToken = expectedOutput[prevOut.length];
                if (nextToken) return [...prevOut, nextToken];
                return prevOut;
              });
              return 0;
            }
            return prevLayer + 1;
          });
          return 0;
        }
        return prev + 1;
      });
    }, 1000 / speed); // Much slower interval for Ollama (visible sequential processing)
    
    return () => clearInterval(ollamaInterval);
  }, [isPlaying, speed]);
  
  const reset = () => {
    setIsPlaying(false);
    setCurrentToken(0);
    setOllamaLayer(0);
    setEdgeLayer(0);
    setOllamaStep(0);
    setEdgeStep(0);
    setOllamaOutput([]);
    setEdgeOutput([]);
    setOllamaTime(0);
    setEdgeTime(0);
  };
  
  // Step forward one step manually
  const stepForward = () => {
    setIsPlaying(false);
    // Step EdgeLLM (faster)
    setEdgeTime(prev => prev + 12.5);
    setEdgeStep((prev) => {
      if (prev >= stepsPerLayer - 1) {
        setEdgeLayer((prevLayer) => {
          if (prevLayer >= totalLayers - 1) {
            setEdgeOutput((prevOut) => {
              const nextToken = expectedOutput[prevOut.length];
              if (nextToken) return [...prevOut, nextToken];
              return prevOut;
            });
            return 0;
          }
          return prevLayer + 1;
        });
        return 0;
      }
      return prev + 1;
    });
    // Step Ollama (slower - only every other step)
    setOllamaTime(prev => prev + 31.25);
    setOllamaStep((prev) => {
      const newStep = prev + 0.4; // Slower progression
      if (newStep >= stepsPerLayer - 1) {
        setOllamaLayer((prevLayer) => {
          if (prevLayer >= totalLayers - 1) {
            setOllamaOutput((prevOut) => {
              const nextToken = expectedOutput[prevOut.length];
              if (nextToken) return [...prevOut, nextToken];
              return prevOut;
            });
            return 0;
          }
          return prevLayer + 1;
        });
        return 0;
      }
      return Math.floor(newStep);
    });
  };
  
  // Step names matching architecture diagram
  const stepNames = ['RMSNorm', 'Q·K·V GEMV', 'Attention', 'Output GEMV', 'RMSNorm₂', 'FFN (gate·up·down)', 'Residual'];
  const getStepName = (step) => stepNames[step] || 'RMSNorm';

  return (
    <div className="min-h-screen bg-[#fafaf9] text-gray-800 p-6">
      {/* Header - umerkhan.io style */}
      <div className="max-w-6xl mx-auto">
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-emerald-50 text-emerald-700 text-xs rounded-full mb-4 border border-emerald-200">
            <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
            Interactive Visualization
          </div>
          <h1 className="text-4xl font-serif font-medium text-gray-900 mb-2">
            LLM Data Flow
          </h1>
          <p className="text-gray-500">Ollama vs EdgeLLM — Side by Side Comparison</p>
          <div className="mt-4 flex justify-center gap-4">
            <div className="inline-flex items-center gap-2 px-3 py-2 bg-orange-50 rounded-full text-sm border border-orange-200">
              <div className="w-2 h-2 bg-orange-500 rounded-full" />
              <span className="text-orange-700 font-medium">Ollama: L{ollamaLayer + 1}/{totalLayers}</span>
              <span className="text-orange-500 text-xs">{getStepName(ollamaStep)}</span>
            </div>
            <div className="inline-flex items-center gap-2 px-3 py-2 bg-emerald-50 rounded-full text-sm border border-emerald-200">
              <div className="w-2 h-2 bg-emerald-500 rounded-full" />
              <span className="text-emerald-700 font-medium">EdgeLLM: L{edgeLayer + 1}/{totalLayers}</span>
              <span className="text-emerald-500 text-xs">{getStepName(edgeStep)}</span>
            </div>
          </div>
        </motion.div>
        
        {/* Controls */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="flex items-center justify-center gap-4 mb-8"
        >
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium transition-all shadow-sm ${
              isPlaying 
                ? 'bg-amber-500 hover:bg-amber-600 text-white' 
                : 'bg-emerald-600 hover:bg-emerald-700 text-white'
            }`}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          
          <button
            onClick={stepForward}
            disabled={isPlaying}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium bg-white border border-gray-200 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-sm"
          >
            <SkipForward className="w-4 h-4" />
            Step
          </button>
          
          <button
            onClick={reset}
            className="flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium bg-white border border-gray-200 hover:bg-gray-50 transition-all shadow-sm"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
          
          <div className="flex items-center gap-1 bg-white border border-gray-200 rounded-lg px-2 py-1.5 shadow-sm">
            <span className="text-sm text-gray-500 mr-1">Speed:</span>
            {[0.15, 0.25, 0.5, 1, 2].map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                  speed === s ? 'bg-gray-900 text-white' : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                {s}x
              </button>
            ))}
          </div>
        </motion.div>
        
        {/* CPU → GPU Data Flow Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl p-5 border border-gray-200 shadow-sm mb-8"
        >
          <h3 className="font-medium text-gray-900 mb-4 flex items-center gap-2">
            <Monitor className="w-4 h-4 text-blue-500" />
            Query Input & Memory Flow
          </h3>
          
          {/* Data Flow Diagram */}
          <div className="flex items-center justify-between gap-4 mb-6">
            {/* User Input */}
            <div className="flex-1 text-center">
              <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mx-auto mb-2">
                <Monitor className="w-6 h-6 text-blue-600" />
              </div>
              <div className="text-sm font-medium text-gray-700">User Query</div>
              <div className="text-xs text-gray-500">"The cat sat on"</div>
            </div>
            
            <ArrowRight className="w-5 h-5 text-gray-300" />
            
            {/* Tokenizer (CPU) */}
            <div className="flex-1 text-center">
              <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center mx-auto mb-2">
                <Cpu className="w-6 h-6 text-purple-600" />
              </div>
              <div className="text-sm font-medium text-gray-700">Tokenizer</div>
              <div className="text-xs text-gray-500">CPU → Token IDs</div>
            </div>
            
            <ArrowRight className="w-5 h-5 text-gray-300" />
            
            {/* PCIe Transfer */}
            <div className="flex-1 text-center">
              <motion.div 
                animate={{ backgroundColor: isPlaying ? ['#dbeafe', '#bfdbfe', '#dbeafe'] : '#f3f4f6' }}
                transition={{ duration: 1, repeat: Infinity }}
                className="w-12 h-12 rounded-xl flex items-center justify-center mx-auto mb-2"
              >
                <Zap className="w-6 h-6 text-blue-600" />
              </motion.div>
              <div className="text-sm font-medium text-gray-700">PCIe Bus</div>
              <div className="text-xs text-gray-500">~16 GB/s</div>
            </div>
            
            <ArrowRight className="w-5 h-5 text-gray-300" />
            
            {/* GPU VRAM */}
            <div className="flex-1 text-center">
              <div className="w-12 h-12 bg-emerald-100 rounded-xl flex items-center justify-center mx-auto mb-2">
                <HardDrive className="w-6 h-6 text-emerald-600" />
              </div>
              <div className="text-sm font-medium text-gray-700">GPU VRAM</div>
              <div className="text-xs text-gray-500">~320 GB/s</div>
            </div>
          </div>

          {/* GPU Memory Layout Comparison */}
          <div className="grid grid-cols-2 gap-4">
            {/* Ollama VRAM */}
            <div className="bg-orange-50 rounded-xl p-4 border border-orange-200">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-2 h-2 bg-orange-500 rounded-full" />
                <span className="text-sm font-medium text-gray-700">Ollama GPU Memory (8+ GB)</span>
              </div>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <div className="h-6 bg-orange-300 rounded flex-[3] flex items-center justify-center text-xs text-orange-800">
                    Weights FP16 (3 GB)
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-6 bg-orange-200 rounded flex-[2] flex items-center justify-center text-xs text-orange-700">
                    KV Cache (2-4 GB)
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-6 bg-orange-100 rounded flex-1 flex items-center justify-center text-xs text-orange-600">
                    Activations
                  </div>
                  <div className="h-6 bg-gray-100 rounded flex-1 flex items-center justify-center text-xs text-gray-500">
                    cuBLAS workspace
                  </div>
                </div>
              </div>
            </div>

            {/* EdgeLLM VRAM */}
            <div className="bg-emerald-50 rounded-xl p-4 border border-emerald-200">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-2 h-2 bg-emerald-500 rounded-full" />
                <span className="text-sm font-medium text-gray-700">EdgeLLM GPU Memory (2 GB)</span>
              </div>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <div className="h-6 bg-emerald-300 rounded flex-1 flex items-center justify-center text-xs text-emerald-800">
                    Weights INT4 (0.75 GB)
                  </div>
                  <div className="h-6 bg-emerald-200 rounded flex-1 flex items-center justify-center text-xs text-emerald-700">
                    Scales (g128)
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-6 bg-emerald-100 rounded flex-1 flex items-center justify-center text-xs text-emerald-600">
                    KV Cache (0.5 GB)
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-6 bg-gray-100 rounded flex-1 flex items-center justify-center text-xs text-gray-500">
                    Activations (minimal)
                  </div>
                  <div className="h-6 bg-emerald-50 rounded flex-1 flex items-center justify-center text-xs text-emerald-500 border border-emerald-200">
                    Free ~0.5 GB
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Memory Bandwidth Explanation */}
          <div className="mt-4 p-3 bg-gray-50 rounded-lg text-xs text-gray-600">
            <strong>Why this matters:</strong> LLM inference is memory-bound. Smaller model (INT4) = less data to read = faster inference.
            <br />EdgeLLM reads 0.75 GB vs Ollama's 3 GB per token generation → <span className="text-emerald-600 font-medium">4x less memory traffic</span>
          </div>
        </motion.div>

        {/* Side-by-Side Comparison */}
        <div className="grid grid-cols-2 gap-6 mb-8">
          {/* OLLAMA SIDE */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-2xl p-5 border border-gray-200 shadow-sm"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-orange-500 rounded-full" />
                <h3 className="font-semibold text-gray-900">Ollama</h3>
                <span className="text-xs text-gray-500">llama.cpp / cuBLAS</span>
              </div>
              <span className="text-xs px-2 py-1 bg-orange-50 text-orange-600 rounded-full">FP16</span>
            </div>
            
            {/* Ollama Flow - uses ollamaStep/ollamaLayer */}
            <div className="space-y-2">
              {['Embedding (FP16)', 'RMSNorm', 'Q·K·V GEMV (1 row/block)', 'Attention', 'Output GEMV', 'FFN (3 GEMV)', 'LM Head'].map((step, i) => (
                <motion.div
                  key={step}
                  animate={{
                    backgroundColor: i <= ollamaStep ? '#fff7ed' : '#f9fafb',
                    borderColor: i === ollamaStep ? '#f97316' : '#e5e7eb',
                    scale: i === ollamaStep && isPlaying ? 1.02 : 1
                  }}
                  transition={{ duration: 0.15 }}
                  className="px-3 py-2 rounded-lg border text-sm flex items-center justify-between"
                >
                  <span className="text-gray-700">{step}</span>
                  {i === ollamaStep && isPlaying && (
                    <motion.div
                      animate={{ opacity: [0.5, 1, 0.5] }}
                      transition={{ duration: 0.5, repeat: Infinity }}
                      className="w-2 h-2 bg-orange-500 rounded-full"
                    />
                  )}
                </motion.div>
              ))}
            </div>
            
            {/* Ollama Row/Block Animation - 1 row at a time (expanded) */}
            <div className="mt-3 p-3 bg-gray-900 rounded-lg">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs text-orange-400 font-medium">GEMV: 1 row/block (slow)</span>
                <span className="text-[10px] text-gray-500">cuBLAS</span>
              </div>
              
              {/* Matrix rows visualization - sequential processing */}
              <div className="space-y-1.5">
                {Array(8).fill(0).map((_, rowIdx) => {
                  const currentActiveRow = ollamaStep % 8;
                  const isActiveRow = rowIdx === currentActiveRow;
                  const isCompletedRow = rowIdx < currentActiveRow;
                  
                  return (
                    <div key={rowIdx} className="flex items-center gap-2">
                      <span className={`text-[10px] w-5 font-mono ${isActiveRow ? 'text-orange-400' : 'text-gray-600'}`}>
                        R{rowIdx + 1}
                      </span>
                      <div className="flex-1 flex gap-0.5">
                        {Array(16).fill(0).map((_, colIdx) => (
                          <div
                            key={colIdx}
                            className={`flex-1 h-3 rounded-sm transition-all duration-300 ${
                              isActiveRow 
                                ? 'bg-orange-500 shadow-[0_0_8px_rgba(249,115,22,0.6)]' 
                                : isCompletedRow 
                                  ? 'bg-orange-600' 
                                  : 'bg-gray-700'
                            }`}
                          />
                        ))}
                      </div>
                      {isActiveRow && (
                        <span className="text-orange-400 text-sm animate-pulse">◀</span>
                      )}
                    </div>
                  );
                })}
              </div>
              
              <div className="mt-3 pt-2 border-t border-gray-700 flex justify-between items-center">
                <span className="text-[10px] text-orange-400">
                  Row {(ollamaStep % 8) + 1} / 8 processing
                </span>
                <span className="text-[10px] text-gray-500 font-mono">
                  Sequential
                </span>
              </div>
            </div>
            
            {/* Ollama Layer Progress */}
            <div className="mt-2 p-2 bg-orange-50/50 rounded-lg">
              <div className="flex justify-between text-xs text-orange-700 mb-1">
                <span>Layer {ollamaLayer + 1}/{totalLayers}</span>
                <span>{Math.round((ollamaLayer / totalLayers) * 100)}%</span>
              </div>
              <div className="h-1.5 bg-orange-200 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-orange-500"
                  animate={{ width: `${((ollamaLayer + 1) / totalLayers) * 100}%` }}
                  transition={{ duration: 0.2 }}
                />
              </div>
            </div>
            
            {/* Ollama Stats */}
            <div className="mt-4 pt-4 border-t border-gray-100 grid grid-cols-3 gap-2 text-center">
              <div>
                <div className="text-lg font-semibold text-orange-600">~32</div>
                <div className="text-xs text-gray-500">tok/s</div>
              </div>
              <div>
                <div className="text-lg font-semibold text-orange-600">3 GB</div>
                <div className="text-xs text-gray-500">Model</div>
              </div>
              <div>
                <div className="text-lg font-semibold text-orange-600">8+ GB</div>
                <div className="text-xs text-gray-500">VRAM</div>
              </div>
            </div>
          </motion.div>

          {/* EDGELLM SIDE */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-2xl p-5 border border-emerald-200 shadow-sm ring-2 ring-emerald-100"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-emerald-500 rounded-full" />
                <h3 className="font-semibold text-gray-900">EdgeLLM</h3>
                <span className="text-xs text-gray-500">Mojo / Custom CUDA</span>
              </div>
              <span className="text-xs px-2 py-1 bg-emerald-50 text-emerald-600 rounded-full">INT4</span>
            </div>
            
            {/* EdgeLLM Flow - uses edgeStep/edgeLayer */}
            <div className="space-y-2">
              {['Embedding (INT8)', 'RMSNorm', 'Q·K·V GEMV (8 rows/block)', 'Attention', 'Output GEMV', 'FFN (Warp Shuffle)', 'LM Head'].map((step, i) => (
                <motion.div
                  key={step}
                  animate={{
                    backgroundColor: i <= edgeStep ? '#ecfdf5' : '#f9fafb',
                    borderColor: i === edgeStep ? '#10b981' : '#e5e7eb',
                    scale: i === edgeStep && isPlaying ? 1.02 : 1
                  }}
                  transition={{ duration: 0.1 }}
                  className="px-3 py-2 rounded-lg border text-sm flex items-center justify-between"
                >
                  <span className="text-gray-700">{step}</span>
                  {i === edgeStep && isPlaying && (
                    <motion.div
                      animate={{ scale: [1, 1.3, 1] }}
                      transition={{ duration: 0.3, repeat: Infinity }}
                      className="w-2 h-2 bg-emerald-500 rounded-full"
                    />
                  )}
                </motion.div>
              ))}
            </div>
            
            {/* EdgeLLM Row/Block Animation - 8 rows in parallel (expanded) */}
            <div className="mt-3 p-3 bg-gray-900 rounded-lg">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs text-emerald-400 font-medium">GEMV: 8 rows/block (fast)</span>
                <span className="text-[10px] text-gray-500">Custom CUDA</span>
              </div>
              
              {/* Matrix rows visualization - all 8 rows process together */}
              <div className="space-y-1.5">
                {Array(8).fill(0).map((_, rowIdx) => {
                  const isActive = isPlaying;
                  
                  return (
                    <div key={rowIdx} className="flex items-center gap-2">
                      <span className={`text-[10px] w-5 font-mono ${isActive ? 'text-emerald-400' : 'text-gray-600'}`}>
                        R{rowIdx + 1}
                      </span>
                      <div className="flex-1 flex gap-0.5">
                        {Array(16).fill(0).map((_, colIdx) => (
                          <div
                            key={colIdx}
                            className={`flex-1 h-3 rounded-sm transition-all duration-200 ${
                              isActive 
                                ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]' 
                                : 'bg-gray-700'
                            }`}
                          />
                        ))}
                      </div>
                      {isActive && (
                        <span className="text-emerald-400 text-sm animate-pulse">◀</span>
                      )}
                    </div>
                  );
                })}
              </div>
              
              <div className="mt-3 pt-2 border-t border-gray-700 flex justify-between items-center">
                <span className="text-[10px] text-emerald-400">
                  All 8 rows in parallel
                </span>
                <span className="text-[10px] text-gray-500 font-mono">
                  Parallel
                </span>
              </div>
            </div>
            
            {/* EdgeLLM Layer Progress */}
            <div className="mt-2 p-2 bg-emerald-50/50 rounded-lg">
              <div className="flex justify-between text-xs text-emerald-700 mb-1">
                <span>Layer {edgeLayer + 1}/{totalLayers}</span>
                <span>{Math.round((edgeLayer / totalLayers) * 100)}%</span>
              </div>
              <div className="h-1.5 bg-emerald-200 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-emerald-500"
                  animate={{ width: `${((edgeLayer + 1) / totalLayers) * 100}%` }}
                  transition={{ duration: 0.1 }}
                />
              </div>
            </div>
            
            {/* EdgeLLM Stats */}
            <div className="mt-4 pt-4 border-t border-emerald-100 grid grid-cols-3 gap-2 text-center">
              <div>
                <div className="text-lg font-semibold text-emerald-600">~80</div>
                <div className="text-xs text-gray-500">tok/s</div>
              </div>
              <div>
                <div className="text-lg font-semibold text-emerald-600">0.75 GB</div>
                <div className="text-xs text-gray-500">Model</div>
              </div>
              <div>
                <div className="text-lg font-semibold text-emerald-600">2 GB</div>
                <div className="text-xs text-gray-500">VRAM</div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Input & Separate Outputs */}
        <div className="grid grid-cols-3 gap-4 mb-8">
          {/* Input Tokens */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm"
          >
            <div className="flex items-center gap-2 mb-3">
              <Database className="w-4 h-4 text-blue-500" />
              <h4 className="font-medium text-gray-900">Input Query</h4>
            </div>
            <div className="flex flex-wrap gap-2">
              {inputTokens.map((token, i) => (
                <span
                  key={i}
                  className="px-2 py-1 rounded text-sm font-mono border bg-blue-50 border-blue-200 text-blue-700"
                >
                  {token}
                </span>
              ))}
            </div>
          </motion.div>

          {/* Ollama Output */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-xl p-4 border border-orange-200 shadow-sm"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-orange-500 rounded-full" />
                <h4 className="font-medium text-gray-900 text-sm">Ollama Output</h4>
              </div>
              <span className="text-xs font-mono text-orange-600 bg-orange-50 px-2 py-0.5 rounded">{ollamaTime.toFixed(0)}ms</span>
            </div>
            <div className="flex flex-wrap gap-2 min-h-[40px] items-center">
              <AnimatePresence mode="popLayout">
                {ollamaOutput.map((token, i) => (
                  <motion.span
                    key={`ollama-${i}-${token}`}
                    initial={{ opacity: 0, scale: 0, y: 20 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0 }}
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                    className="px-3 py-1.5 rounded-lg text-sm font-mono bg-orange-100 border-2 border-orange-300 text-orange-800 font-medium shadow-sm"
                  >
                    {token}
                  </motion.span>
                ))}
              </AnimatePresence>
              {ollamaOutput.length === 0 && (
                <motion.span 
                  animate={{ opacity: [0.4, 0.7, 0.4] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                  className="text-orange-400 text-sm italic"
                >
                  Processing...
                </motion.span>
              )}
            </div>
            <div className="mt-2 text-xs text-gray-500">
              {ollamaOutput.length}/{expectedOutput.length} tokens
            </div>
          </motion.div>

          {/* EdgeLLM Output */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-xl p-4 border border-emerald-200 shadow-sm"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-emerald-500 rounded-full" />
                <h4 className="font-medium text-gray-900 text-sm">EdgeLLM Output</h4>
              </div>
              <span className="text-xs font-mono text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded">{edgeTime.toFixed(0)}ms</span>
            </div>
            <div className="flex flex-wrap gap-2 min-h-[40px] items-center">
              <AnimatePresence mode="popLayout">
                {edgeOutput.map((token, i) => (
                  <motion.span
                    key={`edge-${i}-${token}`}
                    initial={{ opacity: 0, scale: 0, y: 20 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0 }}
                    transition={{ type: "spring", stiffness: 500, damping: 25 }}
                    className="px-3 py-1.5 rounded-lg text-sm font-mono bg-emerald-100 border-2 border-emerald-300 text-emerald-800 font-medium shadow-sm"
                  >
                    {token}
                  </motion.span>
                ))}
              </AnimatePresence>
              {edgeOutput.length === 0 && (
                <motion.span 
                  animate={{ opacity: [0.4, 0.7, 0.4] }}
                  transition={{ duration: 1, repeat: Infinity }}
                  className="text-emerald-400 text-sm italic"
                >
                  Processing...
                </motion.span>
              )}
            </div>
            <div className="mt-2 text-xs text-gray-500">
              {edgeOutput.length}/{expectedOutput.length} tokens
            </div>
          </motion.div>
        </div>
        
        {/* Key Differences */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-xl p-5 border border-gray-200 shadow-sm"
        >
          <h3 className="font-medium text-gray-900 mb-4">Key Optimizations</h3>
          <div className="grid grid-cols-3 gap-4">
            {[
              { label: 'GEMV Strategy', ollama: '1 row/block', edge: '8 rows/block', icon: Cpu },
              { label: 'Memory Access', ollama: 'Standard FP16', edge: 'Vectorized INT4', icon: Database },
              { label: 'Reduction', ollama: 'Shared Memory', edge: 'Warp Shuffle', icon: Zap },
            ].map(({ label, ollama, edge, icon: Icon }) => (
              <div key={label} className="text-center">
                <Icon className="w-5 h-5 text-gray-400 mx-auto mb-2" />
                <div className="text-sm font-medium text-gray-700 mb-2">{label}</div>
                <div className="text-xs text-orange-600 mb-1">{ollama}</div>
                <div className="text-xs text-emerald-600 font-medium">{edge}</div>
              </div>
            ))}
          </div>
        </motion.div>
        
        {/* Footer */}
        <div className="mt-6 text-center text-xs text-gray-400">
          Based on transformer architecture: {totalLayers} layers × 7 GEMV operations per token
        </div>
      </div>
    </div>
  );
}
