/**
 * The first-run chapters decide what is done from the SERVER's flags. A theme
 * boot read that failed (a gateway restart, a dropped link) settles the boot on
 * the browser's render cache instead, and in a browser that never stored
 * `mc-privacy-acked` that cache says "Privacy not acknowledged". Acting on it
 * put the mandatory Privacy chapter back over a user who had already answered
 * it, every time the page loaded during a restart.
 *
 * So the shell hands the chapters a boot that is ready only when the server
 * answered (`themeBootFailed` false). The control case shows the same flags DO
 * open the chapter once the server itself says Privacy is unacknowledged.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'

const themeState = {
  colorTheme: 'kiro',
  theme: 'dark' as const,
  mode: 'dark' as const,
  onboarded: true,
  importOnboarded: true,
  privacyAcked: false,
  privacyAckVersion: null as number | null,
  themeBootReady: true,
  themeBootFailed: true,
  themes: [],
  markOnboarded: vi.fn(),
  markImportOnboarded: vi.fn(),
  markPrivacyAcked: vi.fn(),
  setColorTheme: vi.fn(),
  setMode: vi.fn(),
}

vi.mock('../hooks/useTheme', () => ({
  useTheme: () => themeState,
  ThemeProvider: ({ children }: { children: ReactNode }) => children,
}))

vi.mock('../pages/ChatPage', () => ({ default: () => <div data-testid="chat-page">ChatPage</div> }))
vi.mock('../pages/SystemPage', () => ({ default: () => null }))
vi.mock('../pages/ProjectsPage', () => ({ default: () => null }))
vi.mock('../pages/LogsPage', () => ({ default: () => null }))
vi.mock('../pages/KiroCrewAgentsPage', () => ({ default: () => null }))

import { renderWithProviders } from './helpers'
import App from '../App'

/** The Privacy chapter's own choice; it renders nothing at all while closed. */
function privacyChapterOpen(): boolean {
  return screen.queryByRole('button', { name: /turn off/i }) !== null
}

beforeEach(() => {
  themeState.importOnboarded = true
  themeState.privacyAcked = false
  themeState.themeBootReady = true
  themeState.themeBootFailed = true
})

describe('the Privacy chapter waits for the server', () => {
  it('stays closed while the boot settled without the server', async () => {
    renderWithProviders(<App />, { route: '/chat' })

    await screen.findByTestId('chat-page')
    expect(privacyChapterOpen()).toBe(false)
  })

  it('opens when the server itself says Privacy is unacknowledged', async () => {
    themeState.themeBootFailed = false
    renderWithProviders(<App />, { route: '/chat' })

    await waitFor(() => expect(privacyChapterOpen()).toBe(true))
  })
})
