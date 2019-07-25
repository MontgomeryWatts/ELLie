import discord
import os
import fortune



class ELLie(discord.Client):

	FORTUNE_TELLER = fortune.FortuneTeller('res/fortunes')

	def handle_command(command):
		tokens = command.split(' ')

		if tokens[0] == 'fortune':
			return ELLie.FORTUNE_TELLER.tell()
		else:
			return "No such command."

	async def on_message(self, message):
		if message.content == 'ELL':
			await message.add_reaction('📠')
		elif message.content.startswith(':'):
			res = ELLie.handle_command(message.content[1:].lstrip().rstrip())
			await message.channel.send(res)

ellie = ELLie()
ellie.run(os.environ['DISCORD_TOKEN'])