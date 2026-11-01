/**
 * SettingsTab — Knowledge page ingestion settings.
 *
 * Covers: render, commit-on-blur validation, the extraction-effort select,
 * error banner, and where extraction runs (the provider select and the local
 * endpoint's URL, model and timeout rows). The extraction model / effort / pool-size rows live HERE
 * (this tab is the single home for the Knowledge LLM-pool knobs); ChatPanel
 * carries no Knowledge controls.
 */

// SimpleSelect wraps Radix Select, whose portalled listbox jsdom cannot open;
// the repo's mock (shared with SettingsSelect.test.tsx) makes every picker
// here driveable as real role="option" nodes.
vi.mock('@radix-ui/react-select', async () => await import('./__mocks__/@radix-ui/react-select'))

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'

const { patchConfigMock, kirocrewConfigMock, knowledgeApiMock } = vi.hoisted(() => ({
  patchConfigMock: vi.fn(() => Promise.resolve({})),
  knowledgeApiMock: vi.fn((_path: string): Promise<unknown> =>
    Promise.resolve({ base_url: '', models: [], error: null })),
  kirocrewConfigMock: vi.fn(() => Promise.resolve({
    agent: { model: 'claude-opus-4.8' },
    knowledge: {
      embed_rate_limit: 120,
      extraction_model: '',
      extraction_effort: '',
      extraction_pool_size: 3,
    },
  })),
}))

vi.mock('../api/client', () => ({
  api: {
    kirocrewConfig: kirocrewConfigMock,
    patchConfig: patchConfigMock,
  },
}))

vi.mock('../pages/knowledge/api', () => ({ knowledgeApi: knowledgeApiMock }))

vi.mock('../hooks/useAvailableModels', () => ({
  useAvailableModels: () => [
    { name: 'auto', description: 'Default' },
    { name: 'claude-opus-4.8', description: 'Opus' },
    { name: 'claude-haiku-4.5', description: 'Haiku' },
  ],
}))

import { SettingsTab } from '../pages/knowledge/SettingsTab'

function wrap(opts: { onCommit?: () => void } = {}) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={qc}>
      {/* `onRender` runs in the commit phase, after the DOM is written and before
          passive effects, so a probe there sees exactly what one commit shows. */}
      <React.Profiler id="knowledge-settings" onRender={() => opts.onCommit?.()}>
        <SettingsTab />
      </React.Profiler>
    </QueryClientProvider>,
  )
}

/** Seed the config query with a deep-merged override of the default fixture. */
function seedCfg(knowledge: Record<string, unknown>, agentModel = 'claude-opus-4.8') {
  kirocrewConfigMock.mockImplementation(() =>
    Promise.resolve({
      agent: { model: agentModel },
      knowledge: {
        embed_rate_limit: 120,
        extraction_model: '',
        extraction_effort: '',
        extraction_pool_size: 3,
        ...knowledge,
      },
    }) as never,
  )
}

function rejectOnce(mock: ReturnType<typeof vi.fn>) {
  mock.mockRejectedValueOnce(new Error('fail'))
}

beforeEach(() => {
  patchConfigMock.mockClear()
  kirocrewConfigMock.mockClear()
  knowledgeApiMock.mockClear()
  // The effort row's shared SchemaRestartBadge fetches GET /api/config/schema;
  // without a schema response the badge would honestly render an error notice
  // instead of the pill. Serve the two boot-only knowledge fields as the
  // backend (config schema, restart=True) declares them.
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok: true,
      json: async () => ({
        entries: [
          { path: 'knowledge.extraction_effort', type: 'string', requiresRestart: true },
          { path: 'knowledge.extraction_pool_size', type: 'integer', requiresRestart: true },
        ],
      }),
    })) as unknown as typeof fetch,
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
})

/** Open a SimpleSelect by accessible name; returns its OWN listbox element. */
async function openSelect(label: string): Promise<HTMLElement> {
  const trigger = await screen.findByRole('combobox', { name: label })
  await waitFor(() => expect(trigger).not.toBeDisabled())
  fireEvent.click(trigger)
  // A picker's mocked Content stays mounted while the next picker opens, so
  // several role="listbox" nodes can coexist — the LAST one is the fresh one.
  const boxes = screen.getAllByRole('listbox')
  return boxes[boxes.length - 1]
}

describe('KnowledgeSettingsTab', () => {
  it('renders all 4 settings fields and no removed rows', async () => {
    wrap()
    expect(await screen.findByText('Ingestion Settings')).toBeInTheDocument()
    // Check labels exist
    expect(screen.getByText('Embedding rate limit')).toBeInTheDocument()
    expect(screen.getByText('Extraction model')).toBeInTheDocument()
    expect(screen.getByText('Extraction effort')).toBeInTheDocument()
    expect(screen.getByText('Extraction pool size')).toBeInTheDocument()
    // The two auto-registration knobs are gone with the feature.
    expect(screen.queryByText('Per-source chunk limit')).not.toBeInTheDocument()
    expect(screen.queryByText('Max sources')).not.toBeInTheDocument()
    // The scope cut removed the per-workload fetch effort: URL fetching keeps
    // the provider/model default and has no control anywhere.
    expect(screen.queryByText('URL fetch effort')).not.toBeInTheDocument()
  })

  it('shows the stored numbers in the first commit that enables the inputs', async () => {
    // Both rows are seeded from the served config in the render that first has
    // it, not latched by a later effect: a commit with the inputs enabled and
    // empty is a frame a user can see and a test can land on.
    const commits: { enabled: boolean; rate: string; pool: string }[] = []
    wrap({
      onCommit: () => {
        const rate = screen.queryByRole('spinbutton', { name: 'Embedding rate limit' }) as HTMLInputElement | null
        const pool = screen.queryByRole('spinbutton', { name: 'Extraction pool size' }) as HTMLInputElement | null
        if (rate && pool) commits.push({ enabled: !rate.disabled && !pool.disabled, rate: rate.value, pool: pool.value })
      },
    })
    const rate = await screen.findByRole('spinbutton', { name: 'Embedding rate limit' })
    await waitFor(() => expect(rate).not.toBeDisabled())
    const first = commits.find(c => c.enabled)
    expect(first).toEqual({ enabled: true, rate: '120', pool: '3' })
  })

  it('PATCHes the embedding rate limit on blur with a valid value', async () => {
    wrap()
    // Wait for config to load — inputs get seeded from the mock config
    await waitFor(() => {
      const inputs = document.querySelectorAll('input[type="number"]')
      expect(inputs.length).toBeGreaterThanOrEqual(2)
    })
    const inputs = document.querySelectorAll('input[type="number"]')
    const rateInput = inputs[0] as HTMLInputElement
    await waitFor(() => expect(rateInput.value).toBe('120'))
    fireEvent.change(rateInput, { target: { value: '500' } })
    fireEvent.blur(rateInput)
    await waitFor(() =>
      expect(patchConfigMock).toHaveBeenCalledWith('knowledge.embed_rate_limit', 500),
    )
  })

  it('reverts the embedding rate limit when value is out of range', async () => {
    wrap()
    await waitFor(() => {
      const inputs = document.querySelectorAll('input[type="number"]')
      expect(inputs.length).toBeGreaterThanOrEqual(2)
    })
    const inputs = document.querySelectorAll('input[type="number"]')
    const rateInput = inputs[0] as HTMLInputElement
    await waitFor(() => expect(rateInput.value).toBe('120'))
    fireEvent.change(rateInput, { target: { value: '99999' } })
    fireEvent.blur(rateInput)
    expect(patchConfigMock).not.toHaveBeenCalled()
    expect(rateInput.value).toBe('120')
  })

  it('reverts the embedding rate limit when value is NaN', async () => {
    wrap()
    await waitFor(() => {
      const inputs = document.querySelectorAll('input[type="number"]')
      expect(inputs.length).toBeGreaterThanOrEqual(2)
    })
    const inputs = document.querySelectorAll('input[type="number"]')
    const rateInput = inputs[0] as HTMLInputElement
    await waitFor(() => expect(rateInput.value).toBe('120'))
    fireEvent.change(rateInput, { target: { value: 'abc' } })
    fireEvent.blur(rateInput)
    expect(patchConfigMock).not.toHaveBeenCalled()
    await waitFor(() => expect(rateInput.value).toBe('120'))
  })

  it('shows error banner on save failure', async () => {
    rejectOnce(patchConfigMock)
    wrap()
    await waitFor(() => {
      const inputs = document.querySelectorAll('input[type="number"]')
      expect(inputs.length).toBeGreaterThanOrEqual(2)
    })
    const inputs = document.querySelectorAll('input[type="number"]')
    const rateInput = inputs[0] as HTMLInputElement
    await waitFor(() => expect(rateInput.value).toBe('120'))
    fireEvent.change(rateInput, { target: { value: '300' } })
    fireEvent.blur(rateInput)
    expect(await screen.findByText(/Failed to save knowledge setting/)).toBeInTheDocument()
  })

  describe('extraction model', () => {
    it('lists auto plus the advertised models', async () => {
      wrap()
      const opts = within(await openSelect('Extraction model')).getAllByRole('option')
      expect(opts.map(o => o.textContent)).toEqual([
        'auto (use chat model)',
        'claude-opus-4.8',
        'claude-haiku-4.5',
      ])
    })

    it('PATCHes the model, translating auto to the empty string', async () => {
      seedCfg({ extraction_model: 'claude-haiku-4.5' })
      wrap()
      await openSelect('Extraction model')
      fireEvent.click(screen.getByRole('option', { name: 'auto (use chat model)' }))
      await waitFor(() =>
        expect(patchConfigMock).toHaveBeenCalledWith('knowledge.extraction_model', ''),
      )
    })

    it('keeps a pinned model selectable when the backend stops listing it', async () => {
      seedCfg({ extraction_model: 'claude-opus-4.7-retired' })
      wrap()
      const opts = within(await openSelect('Extraction model')).getAllByRole('option')
      expect(opts.map(o => o.textContent)).toContain('claude-opus-4.7-retired')
      expect(patchConfigMock).not.toHaveBeenCalled()
    })
  })

  describe('extraction effort', () => {
    it('offers the Default option plus every effort level', async () => {
      wrap()
      const opts = within(await openSelect('Extraction effort')).getAllByRole('option')
      // '' is the Default option: with no explicit level the pool runs High.
      expect(opts.map(o => o.textContent)).toEqual([
        'Default',
        'Low',
        'Medium',
        'High',
        'Extra High',
        'Max',
      ])
    })

    it('PATCHes Extraction effort to its own config path', async () => {
      wrap()
      fireEvent.click(within(await openSelect('Extraction effort')).getByRole('option', { name: 'Low' }))
      await waitFor(() =>
        expect(patchConfigMock).toHaveBeenCalledWith('knowledge.extraction_effort', 'low'),
      )
    })

    it('surfaces a failed effort write', async () => {
      rejectOnce(patchConfigMock)
      wrap()
      fireEvent.click(within(await openSelect('Extraction effort')).getByRole('option', { name: 'Low' }))
      expect(await screen.findByText(/Failed to save knowledge setting/)).toBeInTheDocument()
    })

    it('shows the stored effort value once loaded', async () => {
      seedCfg({ extraction_effort: 'xhigh' })
      wrap()
      const trigger = await screen.findByRole('combobox', { name: 'Extraction effort' })
      await waitFor(() => expect(trigger).toHaveTextContent('Extra High'))
    })

    it('shows the default-High hint and the shared restart badge', async () => {
      wrap()
      // findBy*: the hint is only correct once the config query has resolved
      // (before that, the effective model is unknown and the row shows the
      // run-time hint instead).
      expect(
        await screen.findByText(/Default runs High effort until you pick a level/),
      ).toBeInTheDocument()
      // The shared SchemaRestartBadge, same as the pool-size row: the pools
      // read the config once at startup, so a new level reaches workers
      // spawned after a restart only.
      expect(screen.getAllByTestId('restart-required-badge').length).toBeGreaterThan(0)
    })

    it('stays ENABLED with the run-time hint while the effective model is unknown', async () => {
      // Both selectors on 'auto': the model is resolved at run time, so
      // disabling the select would wrongly suggest effort never applies.
      seedCfg({}, 'auto')
      wrap()
      const trigger = await screen.findByRole('combobox', { name: 'Extraction effort' })
      await waitFor(() => expect(trigger).toBeInTheDocument())
      expect(trigger).not.toBeDisabled()
      expect(screen.getByText(/picked at run time/)).toBeInTheDocument()
    })

    // Last in this describe: it re-points the config mock for the rest of the
    // run (beforeEach clears calls, not implementations), so anything after
    // it would inherit the effort-incapable agent model.
    it('disables the effort select with the reasoning-model hint on an effort-incapable model', async () => {
      seedCfg({}, 'claude-haiku-4.5')
      wrap()
      const trigger = await screen.findByRole('combobox', { name: 'Extraction effort' })
      await waitFor(() => expect(trigger).toHaveAttribute('data-disabled'))
      expect(screen.getAllByText(/Applies only on reasoning-capable models/)).not.toHaveLength(0)
      expect(patchConfigMock).not.toHaveBeenCalled()
    })
  })

  describe('extraction pool size', () => {
    /** The pool-size input is the second number input on the tab. */
    async function poolInput() {
      wrap()
      await waitFor(() => {
        const inputs = document.querySelectorAll('input[type="number"]')
        expect(inputs.length).toBeGreaterThanOrEqual(2)
      })
      const inputs = document.querySelectorAll('input[type="number"]')
      return inputs[inputs.length - 1] as HTMLInputElement
    }

    it('PATCHes the pool size on blur with an in-range value', async () => {
      const input = await poolInput()
      await waitFor(() => expect(input.value).toBe('3'))
      fireEvent.change(input, { target: { value: '5' } })
      fireEvent.blur(input)
      await waitFor(() =>
        expect(patchConfigMock).toHaveBeenCalledWith('knowledge.extraction_pool_size', 5),
      )
    })

    it.each([
      ['above the ceiling', '11'],
      ['below the floor', '0'],
      ['not a number', 'abc'],
    ])('reverts the pool size and writes nothing when %s', async (_case, typed) => {
      const input = await poolInput()
      await waitFor(() => expect(input.value).toBe('3'))
      fireEvent.change(input, { target: { value: typed } })
      fireEvent.blur(input)
      expect(patchConfigMock).not.toHaveBeenCalled()
      await waitFor(() => expect(input.value).toBe('3'))
    })

    it('reverts the pool size to the server value when the write fails', async () => {
      rejectOnce(patchConfigMock)
      const input = await poolInput()
      await waitFor(() => expect(input.value).toBe('3'))
      fireEvent.change(input, { target: { value: '6' } })
      fireEvent.blur(input)
      expect(await screen.findByText(/Failed to save knowledge setting/)).toBeInTheDocument()
      await waitFor(() => expect(input.value).toBe('3'))
    })
  })
})

describe('where extraction runs', () => {
  const BASE = 'http://127.0.0.1:13305/api/v1'

  function seedEndpoint(knowledge: Record<string, unknown> = {}) {
    seedCfg({
      extraction_provider: 'openai_compat',
      extraction_base_url: BASE,
      extraction_endpoint_model: '',
      extraction_timeout_secs: 180,
      ...knowledge,
    })
  }

  /** The provider trigger reads 'Agent harness' before the config arrives too,
   *  so wait for a seeded row before asserting what the config decided. */
  async function configLoaded() {
    const rate = await screen.findByRole('spinbutton', { name: 'Embedding rate limit' })
    await waitFor(() => expect(rate).toHaveValue(120))
    await waitFor(() => expect(rate).not.toBeDisabled())
  }

  function serveModels(models: string[], error: string | null = null) {
    knowledgeApiMock.mockImplementation(() => Promise.resolve({ base_url: BASE, models, error }))
  }

  it('defaults to the harness, shows no endpoint rows and asks no server for models', async () => {
    seedCfg({})
    wrap()
    await configLoaded()
    expect(screen.getByRole('combobox', { name: 'Extraction runs on' })).toHaveTextContent('Agent harness')
    expect(screen.queryByText('Endpoint URL')).not.toBeInTheDocument()
    expect(screen.queryByText('Endpoint model')).not.toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: 'Extraction model' })).not.toHaveAttribute('data-disabled')
    expect(knowledgeApiMock).not.toHaveBeenCalled()
  })

  it('reads an unknown provider as the harness, as the backend does', async () => {
    seedCfg({ extraction_provider: 'llamafile' })
    wrap()
    await configLoaded()
    expect(screen.getByRole('combobox', { name: 'Extraction runs on' })).toHaveTextContent('Agent harness')
    expect(screen.queryByText('Endpoint URL')).not.toBeInTheDocument()
  })

  it('PATCHes the provider when the local endpoint is picked', async () => {
    seedCfg({})
    wrap()
    fireEvent.click(within(await openSelect('Extraction runs on')).getByRole('option', { name: 'Local endpoint' }))
    await waitFor(() =>
      expect(patchConfigMock).toHaveBeenCalledWith('knowledge.extraction_provider', 'openai_compat'),
    )
  })

  it('shows the URL read-only and offers the models the server lists', async () => {
    seedEndpoint()
    serveModels(['gemma-4-26B', 'qwen3.6-35B'])
    wrap()
    expect(await screen.findByText(BASE)).toBeInTheDocument()
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument()
    const opts = within(await openSelect('Endpoint model')).getAllByRole('option')
    expect(opts.map(o => o.textContent)).toEqual(['gemma-4-26B', 'qwen3.6-35B'])
    fireEvent.click(screen.getByRole('option', { name: 'qwen3.6-35B' }))
    await waitFor(() =>
      expect(patchConfigMock).toHaveBeenCalledWith('knowledge.extraction_endpoint_model', 'qwen3.6-35B'),
    )
    expect(knowledgeApiMock).toHaveBeenCalledWith('/extraction-endpoint/models')
    // No control on the tab writes where chunks go.
    expect(patchConfigMock).not.toHaveBeenCalledWith('knowledge.extraction_base_url', expect.anything())
  })

  it('keeps a chosen model selectable when the server stops listing it', async () => {
    seedEndpoint({ extraction_endpoint_model: 'old-model' })
    serveModels(['gemma-4-26B'])
    wrap()
    const opts = within(await openSelect('Endpoint model')).getAllByRole('option')
    expect(opts.map(o => o.textContent)).toEqual(['old-model', 'gemma-4-26B'])
    expect(patchConfigMock).not.toHaveBeenCalled()
  })

  it('falls back to a typed name, saying why the list is missing', async () => {
    seedEndpoint()
    serveModels([], 'http://127.0.0.1:13305/api/v1/models: HTTP 404')
    wrap()
    expect(await screen.findByText(/Couldn't read the server's model list \(.*HTTP 404\)/)).toBeInTheDocument()
    const input = screen.getByRole('textbox', { name: 'Endpoint model' })
    await waitFor(() => expect(input).not.toBeDisabled())
    fireEvent.change(input, { target: { value: '  gemma-4-26B  ' } })
    fireEvent.blur(input)
    await waitFor(() =>
      expect(patchConfigMock).toHaveBeenCalledWith('knowledge.extraction_endpoint_model', 'gemma-4-26B'),
    )
  })

  it('writes nothing when the typed name is unchanged', async () => {
    seedEndpoint({ extraction_endpoint_model: 'gemma-4-26B' })
    serveModels([], 'unreachable')
    wrap()
    const input = await screen.findByRole('textbox', { name: 'Endpoint model' })
    await waitFor(() => expect(input).toHaveValue('gemma-4-26B'))
    await waitFor(() => expect(input).not.toBeDisabled())
    fireEvent.change(input, { target: { value: 'gemma-4-26B ' } })
    fireEvent.blur(input)
    // A write runs a microtask later, so a later write is the barrier: once it
    // lands, an earlier one would have too.
    const timeout = screen.getByRole('spinbutton', { name: 'Request timeout (seconds)' })
    fireEvent.change(timeout, { target: { value: '600' } })
    fireEvent.blur(timeout)
    await waitFor(() =>
      expect(patchConfigMock).toHaveBeenCalledWith('knowledge.extraction_timeout_secs', 600),
    )
    expect(patchConfigMock).toHaveBeenCalledTimes(1)
  })

  it('PATCHes an in-range timeout and reverts an out-of-range one', async () => {
    seedEndpoint()
    wrap()
    const input = await screen.findByRole('spinbutton', { name: 'Request timeout (seconds)' })
    await waitFor(() => expect(input).toHaveValue(180))
    fireEvent.change(input, { target: { value: '5' } })
    fireEvent.blur(input)
    await waitFor(() => expect(input).toHaveValue(180))
    fireEvent.change(input, { target: { value: '600' } })
    fireEvent.blur(input)
    await waitFor(() =>
      expect(patchConfigMock).toHaveBeenCalledWith('knowledge.extraction_timeout_secs', 600),
    )
    // The 600 write is the barrier for the refused 5.
    expect(patchConfigMock).toHaveBeenCalledTimes(1)
  })

  it('keeps the harness model and effort rows in place, disabled, while the endpoint runs', async () => {
    seedEndpoint()
    wrap()
    await configLoaded()
    expect(screen.getByRole('combobox', { name: 'Extraction model' })).toHaveAttribute('data-disabled')
    expect(screen.getByRole('combobox', { name: 'Extraction effort' })).toHaveAttribute('data-disabled')
    expect(screen.getAllByText('Not used while extraction runs on a local endpoint.')).toHaveLength(2)
  })
})
