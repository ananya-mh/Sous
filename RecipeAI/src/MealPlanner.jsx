import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  FaUtensils, FaSearch, FaSpinner, FaSignOutAlt, FaArrowLeft, FaClock,
  FaCheckCircle, FaRegCircle, FaShoppingBasket, FaExchangeAlt, FaExclamationTriangle,
  FaClipboardList, FaSync, FaBookOpen, FaTimes
} from 'react-icons/fa'
import './App.css'

const API_BASE_URL = 'http://localhost:5000/api'

function MealPlanner() {
  const [query, setQuery] = useState('')
  const [search, setSearch] = useState(null) // { constraints, unsupported_diet, results }
  const [selected, setSelected] = useState([])
  const [pastedName, setPastedName] = useState('')
  const [pastedText, setPastedText] = useState('')
  const [grocery, setGrocery] = useState(null) // { items, unmerged, substitutions, parser }
  const [loading, setLoading] = useState(null) // 'search' | 'grocery' | null
  const [error, setError] = useState(null)
  const [viewing, setViewing] = useState(null) // { id, recipe?, error? } for the recipe detail modal
  const [user, setUser] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
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

  // Request with the stored token; returns parsed JSON or throws with the API's error detail.
  const apiRequest = async (path, body) => {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      method: body === undefined ? 'GET' : 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`,
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
    if (response.status === 401) {
      handleLogout()
      return null
    }
    const data = await response.json().catch(() => ({}))
    if (!response.ok) {
      throw new Error(data.detail || 'Request failed. Make sure the backend is running.')
    }
    return data
  }
  const apiPost = (path, body) => apiRequest(path, body)

  const handleView = async (id) => {
    setViewing({ id })
    try {
      const recipe = await apiRequest(`/recipes/${id}`)
      if (recipe) setViewing((v) => (v?.id === id ? { id, recipe } : v))
    } catch (err) {
      setViewing((v) => (v?.id === id ? { id, error: err.message } : v))
    }
  }

  const handleSearch = async () => {
    if (!query.trim()) {
      setError('Describe what you want to cook')
      return
    }
    setLoading('search')
    setError(null)
    setGrocery(null)
    setSelected([])
    try {
      const data = await apiPost('/recipes/search', { query })
      if (data) setSearch(data)
    } catch (err) {
      setError(err.message)
      console.error('Search error:', err)
    } finally {
      setLoading(null)
    }
  }

  const toggleRecipe = (id) => {
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]))
  }

  const handleBuildList = async () => {
    if (selected.length === 0 && !pastedText.trim()) {
      setError('Pick at least one recipe or paste your own')
      return
    }
    setLoading('grocery')
    setError(null)
    try {
      const data = await apiPost('/recipes/grocery-list', {
        recipe_ids: selected,
        constraints: search?.constraints ?? {},
        pasted_recipe: pastedText.trim() ? { name: pastedName.trim() || 'My recipe', text: pastedText } : null,
      })
      if (data) setGrocery(data)
    } catch (err) {
      setError(err.message)
      console.error('Grocery list error:', err)
    } finally {
      setLoading(null)
    }
  }

  const handleReset = () => {
    setQuery('')
    setSearch(null)
    setSelected([])
    setPastedName('')
    setPastedText('')
    setGrocery(null)
    setError(null)
  }

  if (!user) {
    return null // Will redirect
  }

  const canBuild = selected.length > 0 || pastedText.trim()

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-amber-50 to-yellow-50">
      <div className="container mx-auto px-4 py-8 max-w-5xl">
        {/* Header */}
        <header className="text-center mb-10">
          <div className="flex justify-between items-center mb-4">
            <button
              onClick={() => navigate('/dashboard')}
              className="px-4 py-2 bg-gray-200 text-gray-700 font-semibold rounded-lg hover:bg-gray-300 focus:outline-none focus:ring-4 focus:ring-gray-300 transition-all flex items-center gap-2"
            >
              <FaArrowLeft />
              Parser
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
              Meal Planner
            </h1>
          </div>
          <p className="text-gray-600 text-lg">Find real recipes, then get one combined grocery list</p>
        </header>

        <div className="bg-white rounded-2xl shadow-2xl p-8 mb-8">
          {/* Search */}
          <label htmlFor="meal-query" className="block text-lg font-semibold text-gray-700 mb-3">
            What do you want to cook?
          </label>
          <div className="flex gap-3">
            <input
              id="meal-query"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="e.g., vegetarian pasta under 30 minutes, no mushrooms"
              className="flex-1 px-5 py-4 border-2 border-gray-200 rounded-xl focus:outline-none focus:border-orange-400 focus:ring-2 focus:ring-orange-200 transition-all text-lg"
              disabled={loading !== null}
            />
            <button
              onClick={handleSearch}
              disabled={loading !== null}
              className="px-8 py-4 bg-gradient-to-r from-orange-500 to-amber-500 text-white font-semibold rounded-xl hover:from-orange-600 hover:to-amber-600 focus:outline-none focus:ring-4 focus:ring-orange-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 flex items-center gap-2"
            >
              {loading === 'search' ? <><FaSpinner className="animate-spin" />Searching...</> : <><FaSearch />Search</>}
            </button>
          </div>
          {error && (
            <div className="mt-4 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg">
              <p className="text-red-700">{error}</p>
            </div>
          )}

          {/* Parsed constraints */}
          {search && (
            <ConstraintChips
              constraints={search.constraints}
              unsupportedDiet={search.unsupported_diet}
              usedFallback={search.parser === 'fallback'}
            />
          )}

          {/* Results */}
          {search && (
            <div className="mt-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <FaClipboardList className="text-orange-600" />
                Pick one or more recipes
              </h2>
              {search.results.length === 0 ? (
                <p className="text-gray-600 p-4 bg-gray-50 rounded-lg">
                  No recipes match all of your constraints. Try loosening the time limit or removing an exclusion.
                </p>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {search.results.map((recipe) => (
                    <RecipeCard
                      key={recipe.id}
                      recipe={recipe}
                      selected={selected.includes(recipe.id)}
                      onToggle={() => toggleRecipe(recipe.id)}
                      onView={() => handleView(recipe.id)}
                    />
                  ))}
                </div>
              )}
            </div>
          )}

          {viewing && (
            <RecipeModal
              viewing={viewing}
              selected={selected.includes(viewing.id)}
              onToggle={() => toggleRecipe(viewing.id)}
              onClose={() => setViewing(null)}
            />
          )}

          {/* Pasted recipe */}
          {search && (
            <div className="mt-8 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border-2 border-blue-200">
              <h2 className="text-xl font-bold text-gray-800 mb-1">Add your own recipe (optional)</h2>
              <p className="text-gray-600 mb-4 text-sm">
                One ingredient per line, with amounts. These are parsed and added up exactly.
              </p>
              <input
                type="text"
                value={pastedName}
                onChange={(e) => setPastedName(e.target.value)}
                placeholder="Recipe name (optional)"
                className="w-full mb-3 px-4 py-3 border-2 border-gray-200 rounded-xl focus:outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-200 transition-all bg-white"
                disabled={loading !== null}
              />
              <textarea
                value={pastedText}
                onChange={(e) => setPastedText(e.target.value)}
                rows={5}
                placeholder={'2 cups chopped tomatoes\n1 tablespoon olive oil\nsalt to taste'}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-200 transition-all bg-white font-mono text-sm"
                disabled={loading !== null}
              />
            </div>
          )}

          {/* Build */}
          {search && (
            <div className="mt-6 flex flex-col sm:flex-row gap-3 justify-center">
              <button
                onClick={handleBuildList}
                disabled={loading !== null || !canBuild}
                className="px-8 py-4 bg-gradient-to-r from-green-500 to-emerald-500 text-white font-semibold rounded-xl hover:from-green-600 hover:to-emerald-600 focus:outline-none focus:ring-4 focus:ring-green-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 flex items-center justify-center gap-2"
              >
                {loading === 'grocery'
                  ? <><FaSpinner className="animate-spin" />Building list...</>
                  : <><FaShoppingBasket />Build grocery list ({selected.length} selected{pastedText.trim() ? ' + yours' : ''})</>}
              </button>
              <button
                onClick={handleReset}
                className="px-6 py-4 bg-gray-200 text-gray-700 font-semibold rounded-xl hover:bg-gray-300 focus:outline-none focus:ring-4 focus:ring-gray-300 transition-all flex items-center justify-center gap-2"
              >
                <FaSync />
                Start Over
              </button>
            </div>
          )}
        </div>

        {grocery && <GroceryResults grocery={grocery} />}
      </div>
    </div>
  )
}

function ConstraintChips({ constraints, unsupportedDiet, usedFallback }) {
  const chips = []
  if (constraints.diet) chips.push({ label: constraints.diet, color: 'bg-green-100 text-green-800' })
  if (constraints.max_minutes) chips.push({ label: `≤ ${constraints.max_minutes} min`, color: 'bg-blue-100 text-blue-800' })
  constraints.include_ingredients.forEach((i) => chips.push({ label: `with ${i}`, color: 'bg-purple-100 text-purple-800' }))
  constraints.exclude_ingredients.forEach((i) => chips.push({ label: `no ${i}`, color: 'bg-red-100 text-red-800' }))
  if (constraints.free_text) chips.push({ label: `"${constraints.free_text}"`, color: 'bg-gray-100 text-gray-700' })

  return (
    <div className="mt-6">
      <p className="text-sm font-semibold text-gray-500 uppercase mb-2">Understood as</p>
      <div className="flex flex-wrap gap-2">
        {chips.map((chip) => (
          <span key={chip.label} className={`px-3 py-1 rounded-full text-sm font-semibold ${chip.color}`}>{chip.label}</span>
        ))}
      </div>
      {usedFallback && (
        <p className="mt-3 text-sm text-yellow-800 bg-yellow-50 p-3 rounded-lg flex items-center gap-2">
          <FaExclamationTriangle />
          The AI query parser was unavailable, so a simpler pattern matcher read your request. Check that
          the constraints above look right.
        </p>
      )}
      {unsupportedDiet && (
        <p className="mt-3 text-sm text-yellow-800 bg-yellow-50 p-3 rounded-lg flex items-center gap-2">
          <FaExclamationTriangle />
          "{unsupportedDiet}" isn't a diet we can filter on yet, so results aren't filtered for it.
        </p>
      )}
    </div>
  )
}

function RecipeCard({ recipe, selected, onToggle, onView }) {
  const more = recipe.ingredient_count - recipe.top_ingredients.length
  return (
    <div
      className={`flex flex-col rounded-xl border-2 transition-all hover:shadow-lg ${
        selected ? 'border-orange-400 bg-orange-50 shadow-md' : 'border-gray-100 bg-white shadow-sm'
      }`}
    >
      <button type="button" onClick={onToggle} aria-pressed={selected} className="text-left p-5 pb-3 flex-1">
        <div className="flex items-start justify-between gap-3 mb-2">
          <h3 className="text-lg font-bold text-gray-800">{recipe.name}</h3>
          <span className={`text-2xl flex-shrink-0 ${selected ? 'text-orange-500' : 'text-gray-300'}`}>
            {selected ? <FaCheckCircle /> : <FaRegCircle />}
          </span>
        </div>
        <p className="text-sm text-gray-500 mb-3 flex items-center gap-1">
          <FaClock />
          {recipe.minutes} min
        </p>
        <p className="text-gray-700 text-sm">
          {recipe.top_ingredients.join(', ')}
          {more > 0 && <span className="text-gray-400"> + {more} more</span>}
        </p>
        {recipe.missing_ingredients.length > 0 && (
          <p className="mt-3 text-sm text-yellow-800 bg-yellow-50 px-3 py-2 rounded-lg">
            Missing: {recipe.missing_ingredients.join(', ')}
          </p>
        )}
        {recipe.diet_notes?.length > 0 && (
          <p className="mt-3 text-sm text-blue-800 bg-blue-50 px-3 py-2 rounded-lg">
            {[...new Set(recipe.diet_notes.map((n) => n.note))].map((note) => note[0].toUpperCase() + note.slice(1)).join('; ')}
            {' '}({recipe.diet_notes.map((n) => n.ingredient).join(', ')})
          </p>
        )}
      </button>
      <button
        type="button"
        onClick={onView}
        className="mx-5 mb-4 self-start text-sm font-semibold text-orange-700 hover:text-orange-800 flex items-center gap-1"
      >
        <FaBookOpen />
        View recipe
      </button>
    </div>
  )
}

function RecipeModal({ viewing, selected, onToggle, onClose }) {
  const closeRef = useRef(null)
  const onCloseRef = useRef(onClose)
  useEffect(() => {
    onCloseRef.current = onClose
  })
  useEffect(() => {
    closeRef.current?.focus() // once, when the modal opens
    const onKey = (e) => e.key === 'Escape' && onCloseRef.current()
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [])

  const { recipe, error } = viewing
  return (
    <div className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label={recipe?.name ?? 'Recipe'}
        className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[85vh] overflow-y-auto p-8"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 mb-4">
          <h2 className="text-2xl font-bold text-gray-800">{recipe?.name ?? 'Loading recipe...'}</h2>
          <button ref={closeRef} type="button" onClick={onClose} aria-label="Close" className="text-gray-400 hover:text-gray-600 text-xl">
            <FaTimes />
          </button>
        </div>

        {error && <p className="text-red-700 bg-red-50 p-3 rounded-lg">{error}</p>}
        {!recipe && !error && <FaSpinner className="animate-spin text-orange-500 text-2xl" />}

        {recipe && (
          <>
            <p className="text-sm text-gray-500 mb-6 flex items-center gap-1">
              <FaClock />
              {recipe.minutes} min
              {recipe.tags.length > 0 && <span className="ml-2">· {recipe.tags.slice(0, 5).join(' · ')}</span>}
            </p>

            <h3 className="text-lg font-bold text-gray-800 mb-1">Ingredients</h3>
            <p className="text-xs text-gray-400 mb-2">Food.com lists amounts without units.</p>
            <ul className="mb-6 space-y-1">
              {recipe.ingredients.map((ing, i) => (
                <li key={`${ing.name}-${i}`} className="text-gray-700">
                  {ing.amount ? <span className="font-semibold">{ing.amount} </span> : null}
                  {ing.name}
                </li>
              ))}
            </ul>

            <h3 className="text-lg font-bold text-gray-800 mb-2">Steps</h3>
            {recipe.steps.length === 0 ? (
              <p className="text-gray-500">No instructions in the dataset for this recipe.</p>
            ) : (
              <ol className="list-decimal pl-6 space-y-2 mb-6">
                {recipe.steps.map((step, i) => <li key={i} className="text-gray-700">{step}</li>)}
              </ol>
            )}

            <button
              type="button"
              onClick={onToggle}
              className={`px-6 py-3 font-semibold rounded-xl flex items-center gap-2 transition-all ${
                selected ? 'bg-orange-100 text-orange-800 hover:bg-orange-200' : 'bg-gradient-to-r from-orange-500 to-amber-500 text-white hover:from-orange-600 hover:to-amber-600'
              }`}
            >
              {selected ? <><FaCheckCircle />Selected for grocery list</> : <><FaRegCircle />Add to grocery list</>}
            </button>
          </>
        )}
      </div>
    </div>
  )
}

function GroceryResults({ grocery }) {
  const substitutions = grocery.substitutions
  return (
    <div className="bg-white rounded-2xl shadow-2xl p-8 mb-8">
      {grocery.parser === 'fallback' && (
        <p className="mb-6 text-sm text-yellow-800 bg-yellow-50 p-3 rounded-lg flex items-center gap-2">
          <FaExclamationTriangle />
          The BERT model isn't loaded, so pasted lines were read with a simpler fallback parser.
        </p>
      )}

      {substitutions.length > 0 && (
        <div className="mb-8 p-6 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl border-2 border-purple-200">
          <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
            <FaExchangeAlt className="text-purple-600" />
            Substitutions
          </h2>
          <div className="space-y-3">
            {substitutions.map((s) => (
              <div key={`${s.original}-${s.constraint}`} className="bg-white p-4 rounded-lg shadow-sm">
                <p className="font-semibold text-gray-800">
                  {s.original} <span className="text-gray-400 font-normal">({s.constraint})</span>
                  {s.result.found && <> → <span className="text-purple-700">{s.result.substitute_item}</span></>}
                </p>
                {s.result.found && (s.result.new_amount || s.result.new_unit) && (
                  <p className="text-sm text-gray-600">Use {s.result.new_amount} {s.result.new_unit}</p>
                )}
                <p className="text-sm text-gray-600">{s.result.reason}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
        <FaShoppingBasket className="text-green-600" />
        Grocery list
      </h2>
      <ul className="divide-y divide-gray-100">
        {grocery.items.map((item) => <GroceryItem key={item.name} item={item} />)}
      </ul>

      {grocery.unmerged.length > 0 && (
        <div className="mt-8">
          <h3 className="text-lg font-bold text-gray-800 mb-1 flex items-center gap-2">
            <FaExclamationTriangle className="text-yellow-600" />
            Check these
          </h3>
          <p className="text-sm text-gray-500 mb-3">These couldn't be combined automatically, so they're listed separately.</p>
          <ul className="divide-y divide-gray-100">
            {grocery.unmerged.map((item, i) => <GroceryItem key={`${item.name}-${i}`} item={item} reason={item.reason} />)}
          </ul>
        </div>
      )}
    </div>
  )
}

function GroceryItem({ item, reason }) {
  return (
    <li className="py-3">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className={`text-lg ${item.amount_known === false ? 'text-gray-800' : 'font-semibold text-gray-800'}`}>
          {item.display}
        </span>
        {item.substitute?.found && (
          <span className="text-sm text-purple-700 bg-purple-50 px-2 py-0.5 rounded">
            swap: {item.substitute.substitute_item}
          </span>
        )}
        {item.diet_note && (
          <span className="text-sm text-blue-800 bg-blue-50 px-2 py-0.5 rounded">{item.diet_note}</span>
        )}
      </div>
      {item.hints?.length > 0 && (
        <p className="text-sm text-gray-500">
          {item.hints.map((h) => `${h.recipe}: ${h.amount} (unit not given)`).join(' · ')}
        </p>
      )}
      <p className="text-xs text-gray-400">
        {reason ? `${reason} · ` : ''}used in {item.recipes.join(', ')}
      </p>
    </li>
  )
}

export default MealPlanner
