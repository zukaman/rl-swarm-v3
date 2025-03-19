import { z } from 'zod'

const leaderboardResponseSchema = z.object({
	leaders: z.array(
		z.object({
			id: z.string(),
			values: z.array(z.object({ x: z.number(), y: z.number() })),
			score: z.number(),
		})
	),
})

export type LeaderboardResponse = z.infer<typeof leaderboardResponseSchema>

type GossipRequest = {
	since: number
}

const gossipResponseSchema = z.object({
	messages: z.array(
		z.object({
			id: z.string(),
			message: z.string(),
			node: z.string(),
		})
	),
	currentRound: z.number(),
	currentStage: z.number(),
})
export type GossipResponse = z.infer<typeof gossipResponseSchema>

export async function getLeaderboard(): Promise<LeaderboardResponse> {
	try {
		const res = await fetch('/api/leaderboard')
		if (!res.ok) {
			throw new Error(`Failed to fetch leaderboard: ${res.statusText}`)
		}

		const json = await res.json()
		const result = leaderboardResponseSchema.parse(json)

		// Truncate to 10, though this should be done server side as well.
		result.leaders = result.leaders.slice(0, 10)

		result.leaders.forEach((leader) => {
			leader.score = parseFloat(leader.score.toFixed(2))
			if (leader.id.toLowerCase() === 'gensyn') {
				leader.id = 'INITIAL PEER'
			}
			leader.values = []
		})

		return result
	} catch (e) {
		if (e instanceof z.ZodError) {
			console.warn('zod error fetching leaderboard details. returning empty leaderboard response.', e)
			return {
				leaders: [],
			}
		} else if (e instanceof Error) {
			console.error('error fetching leaderboard details', e)
			throw new Error(`could not get leaderboard: ${e.message}`)
		} else {
			throw new Error('could not get leaderboard')
		}
	}
}

export async function getGossip(req: GossipRequest): Promise<GossipResponse> {
	try {
		const res = await fetch(`/api/gossip?since_round=${req.since}`)

		if (!res.ok) {
			throw new Error(`failed to fetch gossip: ${res.statusText}`)
		}

		const json = await res.json()

		if (res.status > 499) {
			console.error('5xx error fetching gossip')
			throw new Error('could not get gossip: internal server error')
		}

		const gres = gossipResponseSchema.parse(json)

		gres.messages.forEach((message) => {
			if (message.node.toLocaleLowerCase() === 'gensyn') {
				message.node = 'INITIAL PEER'
			}
		})

		return gres
	} catch (e) {
		if (e instanceof z.ZodError) {
			console.warn('zod error fetching gossip details. returning empty gossip response.', e)
			return {
				messages: [],
				currentRound: -1,
				currentStage: -1,
			}
		} else if (e instanceof Error) {
			console.error('error fetching gossip details', e)
			throw new Error(`could not get gossip: ${e.message}`)
		} else {
			throw new Error('could not get gossip')
		}
	}
}
