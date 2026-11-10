import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { QueryClient, QueryClientProvider, type Query } from '@tanstack/react-query'

/**
 * A theme boot read that FAILED settles the boot on the browser's render cache,
 * which is not the server's answer. Two things follow, and both are pinned here:
 *
 * - `themeBootFailed` says so, because the first-run chapters must not decide
 *   what is done from the cache. A browser with no `mc-privacy-acked` (the ack
 *   was written by another browser, or by the gateway itself) would otherwise
 *   put the mandatory Privacy chapter back over a user who already answered it,
 *   every time the page loads during a gateway restart.
 * - the read is asked again while it is failed, so the page does not stay on
 *   the cache (light theme, first-run flags) until the user reloads.
 */

const themesFn = vi.fn()
const themeDetailFn = vi.fn()
const themeBootFn = vi.fn()
const updateThemeConfigFn = vi.fn()
vi.mock('../api/client', () => ({
  api: {
    themes: () => themesFn(),
    themeDetail: (slug: string) => themeDetailFn(slug),
    themeBoot: () => themeBootFn(),
    updateThemeConfig: (body: unknown) => updateThemeConfigFn(body),
    deleteTheme: vi.fn(),
  },
}))

import { ThemeProvider, useTheme } from '../hooks/useTheme'

const THEME_BOOT_KEY = ['theme-boot']

function mount() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>{children}</ThemeProvider>
    </QueryClientProvider>
  )
  return { queryClient, ...renderHook(() => useTheme(), { wrapper }) }
}

/** The interval the mounted boot query would wait before its next read. */
function bootRefetchInterval(queryClient: QueryClient): unknown {
  const query = queryClient.getQueryCache().find({ queryKey: THEME_BOOT_KEY })
  expect(query, 'no theme-boot query was registered').toBeDefined()
  const option = query!.observers[0]?.options.refetchInterval
  return typeof option === 'function' ? option(query as Query) : option
}

describe('a theme boot read that failed', () => {
  beforeEach(() => {
    localStorage.clear()
    // Import done in this browser, Privacy never acknowledged HERE.
    localStorage.setItem('mc-import-onboarded', '1')
    themesFn.mockResolvedValue({ themes: [] })
    themeDetailFn.mockResolvedValue(null)
    updateThemeConfigFn.mockResolvedValue({})
  })
  afterEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
  })

  it('settles the boot but reports that the server never answered', async () => {
    themeBootFn.mockRejectedValue(new TypeError('Failed to fetch'))
    const { result } = mount()

    await waitFor(() => expect(result.current.themeBootReady).toBe(true))

    expect(result.current.themeBootFailed).toBe(true)
    // The cache's answer, which is exactly why first run must not act on it.
    expect(result.current.privacyAcked).toBe(false)
  })

  it('asks again while failed, and the late answer replaces the cache', async () => {
    themeBootFn.mockRejectedValue(new TypeError('Failed to fetch'))
    const { queryClient, result } = mount()
    await waitFor(() => expect(result.current.themeBootFailed).toBe(true))
    expect(bootRefetchInterval(queryClient)).toBe(30_000)

    themeBootFn.mockResolvedValue({
      mode: 'dark',
      onboarded: true,
      import_onboarded: true,
      privacy_acked: true,
      privacy_ack_version: 2,
    })
    await act(() => queryClient.refetchQueries({ queryKey: THEME_BOOT_KEY }))

    await waitFor(() => expect(result.current.themeBootFailed).toBe(false))
    expect(result.current.privacyAcked).toBe(true)
    expect(result.current.privacyAckVersion).toBe(2)
    expect(bootRefetchInterval(queryClient)).toBe(false)
  })

  it('is not failed when the first read answers', async () => {
    themeBootFn.mockResolvedValue({ onboarded: true, import_onboarded: true, privacy_acked: true })
    const { result } = mount()

    await waitFor(() => expect(result.current.themeBootReady).toBe(true))

    expect(result.current.themeBootFailed).toBe(false)
    expect(result.current.privacyAcked).toBe(true)
  })
})
