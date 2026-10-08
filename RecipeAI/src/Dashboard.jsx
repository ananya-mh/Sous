import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { 
  FaUtensils, FaSearch, FaSpinner, FaSync, FaSignOutAlt,
  FaRuler, FaBox, FaLeaf, FaFileAlt, FaExchangeAlt,
  FaCheckCircle, FaExclamationTriangle, FaRobot, FaBrain, FaChartBar, FaClipboardList
} from 'react-icons/fa'
import './App.css'

function Dashboard() {
  const [recipeLine, setRecipeLine] = useState('')
  const [parsedResult, setParsedResult] = useState(null)
  const [constraint, setConstraint] = useState('')
  const [substitute, setSubstitute] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [user, setUser] = useState(null)
  const navigate = useNavigate()

  const API_BASE_URL = 'http://localhost:5000/api'

  useEffect(() => {
    // Check if user is logged in
    const token = localStorage.getItem('token')
    const userData = localStorage.getItem('user')
    
    if (!token || !userData) {
      navigate('/')
      return
    }

    setUser(JSON.parse(userData))
  }, [navigate])

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    navigate('/')
  }

  const handleParse = async () => {
    if (!recipeLine.trim()) {
      setError('Please enter a recipe line')
      return
    }

    setLoading(true)
    setError(null)
    setSubstitute(null)

    const token = localStorage.getItem('token')

    try {
      const response = await fetch(`${API_BASE_URL}/parse`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({ text: recipeLine }),
      })

      if (response.status === 401) {
        handleLogout()
        return
      }

      if (!response.ok) {
        throw new Error('Failed to parse recipe')
      }

      const data = await response.json()
      setParsedResult(data)
    } catch (err) {
      setError(err.message || 'Failed to parse recipe. Make sure the backend is running.')
      console.error('Parse error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleGetSubstitute = async () => {
    if (!parsedResult?.item) {
      setError('Please parse a recipe first')
      return
    }

    if (!constraint.trim()) {
      setError('Please select a constraint')
      return
    }

    setLoading(true)
    setError(null)

    const token = localStorage.getItem('token')

    try {
      const response = await fetch(`${API_BASE_URL}/substitute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          item: parsedResult.item,
          amount: parsedResult.amount,
          unit: parsedResult.unit,
          constraint: constraint,
        }),
      })

      if (response.status === 401) {
        handleLogout()
        return
      }

      if (!response.ok) {
        throw new Error('Failed to get substitute')
      }

      const data = await response.json()
      setSubstitute(data)
    } catch (err) {
      setError(err.message || 'Failed to get substitute. Make sure the backend is running.')
      console.error('Substitute error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setRecipeLine('')
    setParsedResult(null)
    setConstraint('')
    setSubstitute(null)
    setError(null)
  }

  if (!user) {
    return null // Will redirect
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-amber-50 to-yellow-50">
      <div className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Header with Logout */}
        <header className="text-center mb-12">
          <div className="flex justify-between items-center mb-4">
            <button
              onClick={() => navigate('/planner')}
              className="px-4 py-2 bg-gradient-to-r from-orange-500 to-amber-500 text-white font-semibold rounded-lg hover:from-orange-600 hover:to-amber-600 focus:outline-none focus:ring-4 focus:ring-orange-300 transition-all flex items-center gap-2"
            >
              <FaClipboardList />
              Meal Planner
            </button>
            <div className="flex items-center gap-4">
              <span className="text-gray-600">Welcome, <span className="font-semibold">{user.email}</span></span>
              <button
                onClick={handleLogout}
                className="px-4 py-2 bg-gray-200 text-gray-700 font-semibold rounded-lg hover:bg-gray-300 focus:outline-none focus:ring-4 focus:ring-gray-300 transition-all flex items-center gap-2"
              >
                <FaSignOutAlt />
                Logout
              </button>
            </div>
          </div>
          <div className="flex items-center justify-center gap-3 mb-3">
            <FaUtensils className="text-5xl text-orange-600" />
            <h1 className="text-5xl font-bold bg-gradient-to-r from-orange-600 to-amber-600 bg-clip-text text-transparent">
              Sous
            </h1>
          </div>
          <p className="text-gray-600 text-lg">
            Semantic Ingredient Parsing & Substitution Engine
          </p>
        </header>

        {/* Main Card */}
        <div className="bg-white rounded-2xl shadow-2xl p-8 mb-8">
          {/* Input Section */}
          <div className="mb-8">
            <label htmlFor="recipe-input" className="block text-lg font-semibold text-gray-700 mb-3">
              Enter Recipe Ingredient Line
            </label>
            <div className="flex gap-3">
              <input
                id="recipe-input"
                type="text"
                value={recipeLine}
                onChange={(e) => setRecipeLine(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleParse()}
                placeholder="e.g., 1 cup chopped tomatoes"
                className="flex-1 px-5 py-4 border-2 border-gray-200 rounded-xl focus:outline-none focus:border-orange-400 focus:ring-2 focus:ring-orange-200 transition-all text-lg"
                disabled={loading}
              />
              <button
                onClick={handleParse}
                disabled={loading}
                className="px-8 py-4 bg-gradient-to-r from-orange-500 to-amber-500 text-white font-semibold rounded-xl hover:from-orange-600 hover:to-amber-600 focus:outline-none focus:ring-4 focus:ring-orange-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 flex items-center gap-2"
              >
                {loading ? (
                  <>
                    <FaSpinner className="animate-spin" />
                    Parsing...
                  </>
                ) : (
                  <>
                    <FaSearch />
                    Parse
                  </>
                )}
              </button>
            </div>
            {error && (
              <div className="mt-4 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg">
                <p className="text-red-700">{error}</p>
              </div>
            )}
          </div>

          {/* Parsed Results */}
          {parsedResult && (
            <div className="mb-8 p-6 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border-2 border-green-200">
              <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <FaCheckCircle className="text-green-600" />
                Parsed Results
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <ResultCard label="Amount" value={parsedResult.amount || 'N/A'} icon={<FaRuler />} iconColor="text-blue-600" />
                <ResultCard label="Unit" value={parsedResult.unit || 'N/A'} icon={<FaBox />} iconColor="text-purple-600" />
                <ResultCard label="Item" value={parsedResult.item || 'N/A'} icon={<FaLeaf />} iconColor="text-green-600" />
                <ResultCard label="Descriptor" value={parsedResult.descriptor || 'N/A'} icon={<FaFileAlt />} iconColor="text-indigo-600" />
              </div>
            </div>
          )}

          {/* Substitute Section */}
          {parsedResult && parsedResult.item && (
            <div className="mb-6 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border-2 border-blue-200">
              <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <FaExchangeAlt className="text-blue-600" />
                Find Substitute
              </h2>
              <div className="flex flex-col sm:flex-row gap-4">
                <select
                  value={constraint}
                  onChange={(e) => setConstraint(e.target.value)}
                  className="flex-1 px-5 py-4 border-2 border-gray-200 rounded-xl focus:outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-200 transition-all text-lg bg-white"
                  disabled={loading}
                >
                  <option value="">Select dietary constraint...</option>
                  <option value="vegan">Vegan</option>
                  <option value="keto">Keto</option>
                  <option value="gluten-free">Gluten-Free</option>
                  <option value="dairy-free">Dairy-Free</option>
                  <option value="low-carb">Low-Carb</option>
                  <option value="paleo">Paleo</option>
                </select>
                <button
                  onClick={handleGetSubstitute}
                  disabled={loading || !constraint}
                  className="px-8 py-4 bg-gradient-to-r from-blue-500 to-indigo-500 text-white font-semibold rounded-xl hover:from-blue-600 hover:to-indigo-600 focus:outline-none focus:ring-4 focus:ring-blue-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 flex items-center gap-2"
                >
                  {loading ? (
                    <>
                      <FaSpinner className="animate-spin" />
                      Searching...
                    </>
                  ) : (
                    <>
                      <FaSearch />
                      Find Substitute
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Substitute Result */}
          {substitute && (
            <div className={`p-6 rounded-xl border-2 ${
              substitute.found 
                ? 'bg-gradient-to-r from-purple-50 to-pink-50 border-purple-200' 
                : 'bg-gray-50 border-gray-200'
            }`}>
              {substitute.found ? (
                <>
                  <h3 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <FaCheckCircle className="text-green-600" />
                    Substitute Found!
                  </h3>
                  <div className="space-y-4">
                    <div className="bg-white p-4 rounded-lg shadow-sm">
                      <p className="text-sm font-semibold text-gray-500 mb-1">Substitute Item</p>
                      <p className="text-xl font-bold text-gray-800">{substitute.substitute_item}</p>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="bg-white p-4 rounded-lg shadow-sm">
                        <p className="text-sm font-semibold text-gray-500 mb-1">New Amount</p>
                        <p className="text-xl font-bold text-gray-800">
                          {substitute.new_amount} {substitute.new_unit}
                        </p>
                      </div>
                      <div className="bg-white p-4 rounded-lg shadow-sm">
                        <p className="text-sm font-semibold text-gray-500 mb-1">Reason</p>
                        <p className="text-gray-700">{substitute.reason}</p>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-center py-4 flex items-center justify-center gap-2">
                  <FaExclamationTriangle className="text-yellow-600" />
                  <p className="text-xl text-gray-700">{substitute.reason || 'No suitable substitute found'}</p>
                </div>
              )}
            </div>
          )}

          {/* Reset Button */}
          {(parsedResult || substitute) && (
            <div className="mt-6 text-center">
              <button
                onClick={handleReset}
                className="px-6 py-3 bg-gray-200 text-gray-700 font-semibold rounded-lg hover:bg-gray-300 focus:outline-none focus:ring-4 focus:ring-gray-300 transition-all flex items-center gap-2 mx-auto"
              >
                <FaSync />
                Start Over
              </button>
            </div>
          )}
        </div>

        {/* Info Section */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <InfoCard 
            title="BERT Parser" 
            description="Uses advanced NLP to extract ingredients, amounts, units, and descriptors"
            icon={<FaRobot />}
            iconColor="text-blue-600"
          />
          <InfoCard 
            title="Smart Substitutes" 
            description="AI-powered ingredient substitutions based on dietary constraints"
            icon={<FaBrain />}
            iconColor="text-purple-600"
          />
          <InfoCard 
            title="Accurate Quantities" 
            description="Automatically calculates correct amounts when substituting ingredients"
            icon={<FaChartBar />}
            iconColor="text-green-600"
          />
        </div>
      </div>
    </div>
  )
}

function ResultCard({ label, value, icon, iconColor = "text-orange-600" }) {
  return (
    <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100">
      <div className="flex items-center gap-2 mb-2">
        <span className={`text-xl ${iconColor}`}>{icon}</span>
        <p className="text-sm font-semibold text-gray-500 uppercase">{label}</p>
      </div>
      <p className="text-lg font-bold text-gray-800 break-words">
        {value || <span className="text-gray-400 italic">Not found</span>}
      </p>
    </div>
  )
}

function InfoCard({ title, description, icon, iconColor = "text-orange-600" }) {
  return (
    <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-100 hover:shadow-xl transition-shadow">
      <div className={`text-4xl mb-3 ${iconColor}`}>{icon}</div>
      <h3 className="text-xl font-bold text-gray-800 mb-2">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </div>
  )
}

export default Dashboard
