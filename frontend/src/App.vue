<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { FileText, Search, BookOpen, Loader2 } from 'lucide-vue-next'

const papers = ref([])
const searchQuery = ref('')
const searchResults = ref([])
const selectedPaper = ref(null)
const summary = ref('')
const activeTab = ref('library') // 'library' or 'search'
const isLoading = ref(false)
const error = ref('')

const fetchPapers = async () => {
  try {
    const response = await axios.get('/api/papers')
    papers.value = response.data.papers
  } catch (e) {
    console.error("Error fetching papers:", e)
    error.value = "Failed to load papers."
  }
}

const summarizePaper = async (title) => {
  selectedPaper.value = title
  summary.value = ''
  isLoading.value = true
  error.value = ''
  
  try {
    const response = await axios.post('/api/summarize', { title })
    summary.value = response.data.summary
  } catch (e) {
    console.error("Error summarizing:", e)
    error.value = "Failed to generate summary."
  } finally {
    isLoading.value = false
  }
}

const performSearch = async () => {
  if (!searchQuery.value.trim()) return
  
  isLoading.value = true
  searchResults.value = []
  error.value = ''
  
  try {
    const response = await axios.post('/api/search', { query: searchQuery.value })
    // DuckDuckGo returns a string usually, or we might need to parse it in backend.
    // Assuming backend returns { results: string }
    searchResults.value = response.data.results
  } catch (e) {
    console.error("Error searching:", e)
    error.value = "Search failed."
  } finally {
    isLoading.value = false
  }
}

onMounted(() => {
  fetchPapers()
  // Poll for new papers every 5 seconds
  setInterval(fetchPapers, 5000)
})
</script>

<template>
  <div class="min-h-screen bg-gray-50 text-gray-900 font-sans flex">
    <!-- Sidebar -->
    <aside class="w-64 bg-white border-r border-gray-200 flex flex-col">
      <div class="p-6 border-b border-gray-200">
        <h1 class="text-xl font-bold text-blue-600 flex items-center gap-2">
          <BookOpen class="w-6 h-6" />
          Research Agent
        </h1>
      </div>
      
      <nav class="flex-1 p-4 space-y-2">
        <button 
          @click="activeTab = 'library'"
          :class="['w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors', activeTab === 'library' ? 'bg-blue-50 text-blue-700' : 'hover:bg-gray-100 text-gray-700']"
        >
          <FileText class="w-5 h-5" />
          Library
        </button>
        <button 
          @click="activeTab = 'search'"
          :class="['w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors', activeTab === 'search' ? 'bg-blue-50 text-blue-700' : 'hover:bg-gray-100 text-gray-700']"
        >
          <Search class="w-5 h-5" />
          Web Search
        </button>
      </nav>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 flex flex-col h-screen overflow-hidden">
      <!-- Library View -->
      <div v-if="activeTab === 'library'" class="flex-1 flex overflow-hidden">
        <!-- Paper List -->
        <div class="w-1/3 border-r border-gray-200 bg-white overflow-y-auto p-4">
          <h2 class="text-lg font-semibold mb-4 px-2">Available Papers</h2>
          <div v-if="papers.length === 0" class="text-gray-500 text-center py-8">
            No papers found. <br> Add PDFs to <code>research_agent/papers</code>
          </div>
          <ul class="space-y-2">
            <li v-for="paper in papers" :key="paper">
              <button 
                @click="summarizePaper(paper)"
                :class="['w-full text-left p-3 rounded-lg border transition-all', selectedPaper === paper ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-500' : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50']"
              >
                <div class="font-medium truncate" :title="paper">{{ paper }}</div>
              </button>
            </li>
          </ul>
        </div>

        <!-- Summary View -->
        <div class="flex-1 bg-gray-50 overflow-y-auto p-8">
          <div v-if="selectedPaper" class="max-w-3xl mx-auto">
            <h2 class="text-2xl font-bold mb-6 text-gray-800">
              <a :href="`/api/pdfs/${selectedPaper}`" target="_blank" class="hover:text-blue-600 hover:underline">
                {{ selectedPaper }}
              </a>
            </h2>
            
            <div v-if="isLoading" class="flex items-center justify-center py-12">
              <Loader2 class="w-8 h-8 animate-spin text-blue-600" />
              <span class="ml-3 text-gray-600">Generating summary...</span>
            </div>
            
            <div v-else-if="error" class="p-4 bg-red-50 text-red-700 rounded-lg border border-red-200">
              {{ error }}
            </div>
            
            <div v-else class="bg-white p-8 rounded-xl shadow-sm border border-gray-200 prose max-w-none">
              <h3 class="text-lg font-semibold text-gray-700 mb-4">Summary</h3>
              <div class="whitespace-pre-wrap leading-relaxed text-gray-600">{{ summary }}</div>
            </div>
          </div>
          
          <div v-else class="h-full flex flex-col items-center justify-center text-gray-400">
            <FileText class="w-16 h-16 mb-4 opacity-50" />
            <p class="text-lg">Select a paper to view its summary</p>
          </div>
        </div>
      </div>

      <!-- Search View -->
      <div v-if="activeTab === 'search'" class="flex-1 overflow-y-auto p-8">
        <div class="max-w-3xl mx-auto">
          <h2 class="text-2xl font-bold mb-6">Find Related Papers</h2>
          
          <div class="flex gap-2 mb-8">
            <input 
              v-model="searchQuery" 
              @keyup.enter="performSearch"
              type="text" 
              placeholder="Enter topic or keywords..." 
              class="flex-1 p-4 rounded-xl border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none shadow-sm"
            />
            <button 
              @click="performSearch"
              :disabled="isLoading || !searchQuery"
              class="px-6 py-4 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm flex items-center gap-2"
            >
              <Search class="w-5 h-5" />
              Search
            </button>
          </div>

          <div v-if="isLoading" class="flex justify-center py-12">
            <Loader2 class="w-8 h-8 animate-spin text-blue-600" />
          </div>

          <div v-else-if="searchResults.length > 0 || typeof searchResults === 'string'" class="space-y-4">
             <!-- Handle string result from DuckDuckGo -->
             <div v-if="typeof searchResults === 'string'" class="bg-white p-6 rounded-xl shadow-sm border border-gray-200 whitespace-pre-wrap">
               {{ searchResults }}
             </div>
             <!-- Handle array result if we parse it later -->
             <div v-else v-for="(result, idx) in searchResults" :key="idx" class="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
               {{ result }}
             </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>
