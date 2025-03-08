import { describe, it, expect, vi, beforeEach } from 'vitest'
import { getLeaderboard } from './swarm.api'

describe('getLeaderboard', () => {
    beforeEach(() => {
        vi.clearAllMocks()
    });

    it('should truncate leaders to 10', async () => {
        vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
            json: () => Promise.resolve({
                leaders: new Array(20).fill(undefined).map((_, i) => {
                    return {
                        id: `node-${i}`,
                        score: 20 - i,
                        values: [{ x: 0, y: 20 - i }],
                    }
                })
            })
        })))

        const result = await getLeaderboard()
        expect(result.leaders).toHaveLength(10)

        vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
            json: () => Promise.resolve({
                leaders: new Array(5).fill(undefined).map((_, i) => {
                    return {
                        id: `node-${i}`,
                        score: 20 - i,
                        values: [{ x: 0, y: 20 - i }],
                    }
                })
            })
        })))

        const resultShort = await getLeaderboard()
        expect(resultShort.leaders).toHaveLength(5)

    });
});
