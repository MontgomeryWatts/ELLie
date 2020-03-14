import discord
import os

class ELLie(discord.Client):
	async def on_message(self, message):
		if message.content == 'ELL':
			await message.add_reaction('📠')

ellie = ELLie()
ellie.run(os.environ['DISCORD_TOKEN'])