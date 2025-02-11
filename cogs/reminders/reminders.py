import logging
from datetime import datetime
from dateutil.rrule import rrulestr

import discord
from discord import Interaction, app_commands
from discord.ext import commands, tasks

from common import dataio
from common.utils import fuzzy

from cogs.assistant import assistant as ascog

logger = logging.getLogger(f'MVRIA.{__name__.split(".")[-1]}')

class UserReminder:
    def __init__(self, id: int, user: discord.User | discord.Member, content: str, remind_at: datetime, 
                 is_recurring: bool = False, rrule: str | None = None, end_date: datetime | None = None):
        self._id = id
        self.user = user
        self.content = content
        self.remind_at = remind_at
        self.is_recurring = is_recurring
        self.rrule = rrule
        self.end_date = end_date

    def __str__(self):
        return f'{self.user} - {self.content} - {self.remind_at}'

    def __repr__(self):
        return f'<UserReminder user={self.user} content={self.content} remind_at={self.remind_at}>'

    def to_dict(self):
        return {
            'id': self._id,
            'user': self.user.id,
            'content': self.content,
            'remind_at': self.remind_at.timestamp(),
            'is_recurring': int(self.is_recurring),
            'rrule': self.rrule,
            'end_date': self.end_date.timestamp() if self.end_date else None
        }

# COG ==========================================================================
class Reminders(commands.Cog):
    """Système de rappels compatible avec l'assistant (assistant.py)"""
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.data = dataio.get_instance(self)
        
        # Update table schema to include recurrence columns
        reminders_table = dataio.TableBuilder(
            '''CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                content TEXT,
                remind_at INTEGER,
                is_recurring INTEGER DEFAULT 0,
                rrule TEXT,
                end_date INTEGER
            )'''
        )
        self.data.map_builders('global', reminders_table)
        
        self.check_reminders.start()
        
        self.GPT_TOOLS = [
            ascog.GPTTool(name='add_simple_reminder', 
                                  description="Créer un rappel simple pour l'utilisateur actuel. Faire attention à la date actuelle.",
                                  properties={'content': {'type': 'string', 'description': "Contenu du rappel"},
                                              'remind_at': {'type': 'string', 'description': "Date et heure du rappel au format ISO 8601 (ex: 2024-12-31T23:59:59)"}},
                                  function=self._tool_add_simple_reminder,
                                  footer="<:bell_icon:1338660193466191962> Ajout d'un rappel"),
            ascog.GPTTool(name='add_recurring_reminder',
                                    description="Créer un rappel récurrent pour l'utilisateur actuel. Faire attention à la date actuelle.",
                                    properties={'content': {'type': 'string', 'description': "Contenu du rappel"},
                                                'remind_at': {'type': 'string', 'description': "Date et heure du premier rappel au format ISO 8601 (ex: 2024-12-31T23:59:59)"},
                                                'recurrence': {'type': 'string', 'description': "Fréquence de récurrence (heure, jour, semaine, mois, année)", 'enum': ["Chaque heure", "Chaque jour", "Chaque semaine", "Chaque mois", "Chaque année"]},
                                                'end_date': {'type': ['string', 'null'], 'description': "Date de fin de récurrence au format ISO 8601 (optionnel)"}},
                                    function=self._tool_add_recurring_reminder,
                                    footer="<:mult_bells_icon:1338660033289781358> Ajout d'un rappel récurrent"),
            ascog.GPTTool(name='list_reminders',
                                    description="Lister les rappels enregistrés pour l'utilisateur actuel. Renvoie un dictionnaire contenant ID, contenu et date(s) du rappel.",
                                    properties={},
                                    function=self._tool_list_reminders,
                                    footer="<:look_icon:1338658889243164712> Consultation des rappels"),
            ascog.GPTTool(name='remove_reminder',
                                    description="Supprimer un rappel enregistré pour l'utilisateur actuel.",
                                    properties={'reminder_id': {'type': 'integer', 'description': "Identifiant du rappel à supprimer"}},
                                    function=self._tool_remove_reminder,
                                    footer="<:trash_icon:1338658009466929152> Suppression d'un rappel")
        ]
        
    def cog_unload(self):
        self.check_reminders.stop()
        self.data.close_all()
        
    # Loop ---------------------------------------------------------------------
    
    @tasks.loop(seconds=10)
    async def check_reminders(self):
        now = datetime.now()
        reminders = self.get_all_reminders()
        for reminder in reminders:
            if reminder.remind_at <= now:
                await self.send_reminder(reminder)
                if reminder.is_recurring and reminder.rrule:
                    # Calcul de la prochaine occurrence
                    try:
                        rule = rrulestr(reminder.rrule, dtstart=reminder.remind_at)
                        next_occurrence = rule.after(reminder.remind_at)
                    except Exception:
                        next_occurrence = None
                    # Vérifier la date de fin si spécifiée
                    if next_occurrence and (not reminder.end_date or next_occurrence <= reminder.end_date):
                        reminder.remind_at = next_occurrence
                        self.data.get('global').execute('UPDATE reminders SET remind_at = ? WHERE id = ?', 
                                                          next_occurrence.timestamp(), reminder._id)
                    else:
                        self.remove_reminder(reminder._id)
                else:
                    self.remove_reminder(reminder._id)
                    
    # Utils --------------------------------------------------------------------
    
    def extract_time_from_string(self, string: str) -> datetime | None:
        """Extrait une date d'une chaîne de caractères
        
        :param string: Chaîne de caractères à analyser
        :return: Date extraite ou None
        """
        now = datetime.now()
        formats = [
            '%d/%m/%Y %H:%M',
            '%d/%m/%Y %H',
            '%d/%m/%Y',
            '%d/%m %H:%M',
            '%d/%m',
            '%H:%M'
        ]
        extracted = None
        for format in formats:
            try:
                extracted = datetime.strptime(string, format)
                break
            except ValueError:
                pass
        if extracted:
            # On corrige les dates incomplètes
            if extracted.year == 1900:
                extracted = extracted.replace(year=now.year)
            if extracted.month == 1:
                extracted = extracted.replace(month=now.month)
            if extracted.day == 1:
                extracted = extracted.replace(day=now.day)
        else:
            return now
        return extracted
    
    # Gestions des rappels -----------------------------------------------------
    
    def get_all_reminders(self) -> list[UserReminder]:
        r = self.data.get('global').fetchall('SELECT * FROM reminders')
        reminders = []
        for row in r:
            user = self.bot.get_user(row['user_id'])
            if not isinstance(user, discord.User | discord.Member):
                continue
            remind_at = datetime.fromtimestamp(row['remind_at'])
            reminders.append(UserReminder(row['id'], user, row['content'], remind_at, bool(row['is_recurring']), row['rrule'], datetime.fromtimestamp(row['end_date']) if row['end_date'] else None))
        return reminders
    
    def get_reminders(self, user: discord.User | discord.Member) -> list[UserReminder]:
        r = self.data.get('global').fetchall('SELECT * FROM reminders WHERE user_id = ?', user.id)
        reminders = []
        for row in r:
            remind_at = datetime.fromtimestamp(row['remind_at'])
            reminders.append(UserReminder(row['id'], user, row['content'], remind_at, bool(row['is_recurring']), row['rrule'], datetime.fromtimestamp(row['end_date']) if row['end_date'] else None))
        return reminders
    
    def get_reminder(self, reminder_id: int) -> UserReminder | None:
        r = self.data.get('global').fetchone('SELECT * FROM reminders WHERE id = ?', reminder_id)
        if not r:
            return None
        user = self.bot.get_user(r['user_id'])
        if not isinstance(user, discord.User | discord.Member):
            return None
        remind_at = datetime.fromtimestamp(r['remind_at'])
        return UserReminder(r['id'], user, r['content'], remind_at, bool(r['is_recurring']), r['rrule'], datetime.fromtimestamp(r['end_date']) if r['end_date'] else None)
    
    def add_reminder(self, user: discord.User | discord.Member, content: str, remind_at: datetime,
                     is_recurring: bool = False, rrule: str | None = None, end_date: datetime | None = None):
        self.data.get('global').execute(
            '''INSERT OR REPLACE INTO reminders (user_id, content, remind_at, is_recurring, rrule, end_date)
               VALUES (?, ?, ?, ?, ?, ?)''',
            user.id, content, remind_at.timestamp(), int(is_recurring), rrule, 
            end_date.timestamp() if end_date else None
        )
        
    def remove_reminder(self, reminder_id: int):
        self.data.get('global').execute('DELETE FROM reminders WHERE id = ?', reminder_id)
        
    # Remplacer la méthode qui interprète la récurrence par de simples options
    def parse_natural_recurrence(self, text: str) -> str | None:
        text = text.lower().strip()
        mapping = {
            "chaque heure": "FREQ=HOURLY",
            "chaque jour": "FREQ=DAILY",
            "chaque semaine": "FREQ=WEEKLY",
            "chaque mois": "FREQ=MONTHLY",
            "chaque année": "FREQ=YEARLY"
        }
        return mapping.get(text)

    # Envoi de rappels ---------------------------------------------------------
    
    async def send_reminder(self, reminder: UserReminder):
        if reminder.is_recurring:
            text = f"*{reminder.content}*\n-# <:bell_ring_icon:1338660290077655141> <t:{int(reminder.remind_at.timestamp())}:R> <:loop_icon:1338661846626074655> · {reminder.user.mention}"
        else:
            text = f"*{reminder.content}*\n-# <:bell_ring_icon:1338660290077655141> <t:{int(reminder.remind_at.timestamp())}:R> · {reminder.user.mention}"
        try:
            dm = await reminder.user.create_dm()
        except discord.Forbidden:
            logger.warning(f"Impossible d'envoyer un message privé à {reminder.user} pour le rappel '{reminder.content}'")
            return
        await dm.send(text)
        
    # Commandes ----------------------------------------------------------------
    
    reminder_group = app_commands.Group(name='reminders', description='Gestion de vos rappels')
    
    @reminder_group.command(name='list')
    async def cmd_list_reminders(self, interaction: Interaction):
        """Liste vos rappels enregistrés"""
        user = interaction.user
        reminders = self.get_reminders(user)
        if not reminders:
            await interaction.response.send_message("**Aucun rappel** • Vous n'avez aucun rappel enregistré.", ephemeral=True)
            return
        reminders.sort(key=lambda r: r.remind_at)
        header = "## <:bell_icon:1338660193466191962> __Rappels enregistrés :__\n"
        lines = []
        for reminder in reminders:
            recur_tag = "[R] " if reminder.is_recurring else ""
            time_format = f"<t:{int(reminder.remind_at.timestamp())}:F>"
            lines.append(f"• {recur_tag}{time_format} → `{reminder.content}`")
        await interaction.response.send_message(header + "\n".join(lines), ephemeral=True)
        
    reminder_add_subgroup = app_commands.Group(name='add', description='Ajouter manuellement un rappel', parent=reminder_group)
        
    @reminder_add_subgroup.command(name='simple')
    @app_commands.rename(content='contenu', remind_at='date')
    async def cmd_add_reminder(self, interaction: Interaction, content: str, remind_at: str):
        """Ajouter un rappel

        :param content: Contenu du rappel
        :param remind_at: Date et heure du rappel
        """
        user = interaction.user
        time = self.extract_time_from_string(remind_at) 
        if not time:
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Erreur** × Date et heure non reconnues.", ephemeral=True)
        self.add_reminder(user, content, time)
        await interaction.response.send_message(f"<:bell_icon:1338660193466191962> **Rappel enregistré** • Le rappel sera envoyé en MP <t:{int(time.timestamp())}:F>.", ephemeral=True)
    
    @reminder_add_subgroup.command(name='recurring')
    @app_commands.rename(content='contenu', remind_at='date', recurrence='répétition', end_date='fin')
    async def cmd_add_recurring_reminder(self, interaction: Interaction, content: str, remind_at: str, recurrence: str, end_date: str = ""):
        """Ajouter un rappel récurrent
        
        :param content: Contenu du rappel
        :param remind_at: Date et heure du premier rappel
        :param recurrence: Fréquence de récurrence (chaque jour, semaine, mois, année)
        :param end_date: Date de fin de récurrence (optionnel)
        """
        user = interaction.user
        time = self.extract_time_from_string(remind_at)
        if not time:
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Erreur** × Date non reconnue.", ephemeral=True)
        fin = self.extract_time_from_string(end_date) if end_date else None
        parsed_rrule = self.parse_natural_recurrence(recurrence.lower())
        if not parsed_rrule:
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Erreur** × Récurrence non reconnue. Utilisez par exemple 'tous les jours à 22h'.", ephemeral=True)
        self.add_reminder(user, content, time, True, parsed_rrule, fin)
        await interaction.response.send_message(f"<:mult_bells_icon:1338660033289781358> **Rappel récurrent enregistré** • Premier envoi prévu en MP <t:{int(time.timestamp())}:F>.", ephemeral=True)
    
    @cmd_add_reminder.autocomplete('remind_at')
    @cmd_add_recurring_reminder.autocomplete('remind_at')
    @cmd_add_recurring_reminder.autocomplete('end_date')
    async def autocomplete_time(self, interaction: Interaction, current: str):
        date = self.extract_time_from_string(current)
        if not date:
            return []
        return [app_commands.Choice(name=date.strftime('%d/%m/%Y %H:%M'), value=date.strftime('%d/%m/%Y %H:%M'))]
    
    # Mise à jour de l'autocomplete pour le paramètre "répétition" avec des options simples
    @cmd_add_recurring_reminder.autocomplete('recurrence')
    async def autocomplete_recurrence(self, interaction: Interaction, current: str):
        options = ["Chaque heure", "Chaque jour", "Chaque semaine", "Chaque mois", "Chaque année"]
        return [app_commands.Choice(name=opt, value=opt) for opt in options if current.lower() in opt.lower()]

    @reminder_group.command(name='remove')
    @app_commands.rename(reminder_id='identifiant')
    async def cmd_remove_reminder(self, interaction: Interaction, reminder_id: int):
        """Supprimer un rappel

        :param reminder_id: Identifiant du rappel
        """
        user = interaction.user
        reminder = self.get_reminder(reminder_id)
        if not reminder:
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Erreur** × Rappel introuvable.", ephemeral=True)
        if reminder.user != user:
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Erreur** × Vous n'êtes pas l'auteur de ce rappel.", ephemeral=True)
        self.remove_reminder(reminder_id)
        await interaction.response.send_message(f"<:trash_icon:1338658009466929152> **Rappel supprimé** • Le rappel pour le <t:{int(reminder.remind_at.timestamp())}:F> a été supprimé.", ephemeral=True)
        
    @cmd_remove_reminder.autocomplete('reminder_id')
    async def autocomplete_reminder_id(self, interaction: Interaction, current: str):
        user = interaction.user
        reminders = self.get_reminders(user)
        f = fuzzy.finder(current, reminders, key=lambda r: r.content)
        return [app_commands.Choice(name=f'{r._id} · {r.content[:50]}', value=r._id) for r in f]
    
    # OUTILS et intégration avec l'assistant -----------------------------------
    
    def _tool_add_simple_reminder(self, tool_call: ascog.ToolCall, interaction: ascog.InteractionGroup) -> ascog.ToolAnswerMessage:
        user = interaction.fetch_author()
        if not user:
            return ascog.ToolAnswerMessage({'error': 'Utilisateur non trouvé.'}, tool_call.data['id'])
        
        content = tool_call.arguments.get('content', '')
        if not content:
            return ascog.ToolAnswerMessage({'error': 'Contenu du rappel vide.'}, tool_call.data['id'])
        
        remind_at_iso = tool_call.arguments.get('remind_at', '')
        try:
            remind_at = datetime.fromisoformat(remind_at_iso)
        except ValueError:
            return ascog.ToolAnswerMessage({'error': 'Date et heure non reconnues.'}, tool_call.data['id'])
        
        self.add_reminder(user, content, remind_at)
        return ascog.ToolAnswerMessage({'user': user.id, 'content': content, 'remind_at': remind_at.strftime('%d/%m/%Y %H:%M')}, tool_call.data['id'])
    
    def _tool_add_recurring_reminder(self, tool_call: ascog.ToolCall, interaction: ascog.InteractionGroup) -> ascog.ToolAnswerMessage:
        user = interaction.fetch_author()
        if not user:
            return ascog.ToolAnswerMessage({'error': 'Utilisateur non trouvé.'}, tool_call.data['id'])
        
        content = tool_call.arguments.get('content', '')
        if not content:
            return ascog.ToolAnswerMessage({'error': 'Contenu du rappel vide.'}, tool_call.data['id'])
        
        remind_at_iso = tool_call.arguments.get('remind_at', '')
        try:
            remind_at = datetime.fromisoformat(remind_at_iso)
        except ValueError:
            return ascog.ToolAnswerMessage({'error': 'Date et heure non reconnues.'}, tool_call.data['id'])
        
        recurrence = tool_call.arguments.get('recurrence', '')
        parsed_rrule = self.parse_natural_recurrence(recurrence)
        if not parsed_rrule:
            return ascog.ToolAnswerMessage({'error': 'Récurrence non reconnue.'}, tool_call.data['id'])
        
        end_date_iso = tool_call.arguments.get('end_date', '')
        fin = self.extract_time_from_string(end_date_iso) if end_date_iso else None
        
        self.add_reminder(user, content, remind_at, True, parsed_rrule, fin)
        return ascog.ToolAnswerMessage({'user': user.id, 'content': content, 'remind_at': remind_at.strftime('%d/%m/%Y %H:%M'), 'recurrence': recurrence, 'end_date': end_date_iso}, tool_call.data['id'])
    
    def _tool_list_reminders(self, tool_call: ascog.ToolCall, interaction: ascog.InteractionGroup) -> ascog.ToolAnswerMessage:
        user = interaction.fetch_author()
        if not user:
            return ascog.ToolAnswerMessage({'error': 'Utilisateur non trouvé.'}, tool_call.data['id'])
        
        reminders = self.get_reminders(user)
        if not reminders:
            return ascog.ToolAnswerMessage({'message': 'Aucun rappel enregistré.'}, tool_call.data['id'])
        
        data = [{'id': r._id, 'content': r.content, 'remind_at': r.remind_at.strftime('%d/%m/%Y %H:%M')} for r in reminders]
        return ascog.ToolAnswerMessage({'reminders': data}, tool_call.data['id'])
    
    def _tool_remove_reminder(self, tool_call: ascog.ToolCall, interaction: ascog.InteractionGroup) -> ascog.ToolAnswerMessage:
        user = interaction.fetch_author()
        if not user:
            return ascog.ToolAnswerMessage({'error': 'Utilisateur non trouvé.'}, tool_call.data['id'])
        
        reminder_id = tool_call.arguments.get('reminder_id', 0)
        reminder = self.get_reminder(reminder_id)
        if not reminder:
            return ascog.ToolAnswerMessage({'error': 'Rappel introuvable.'}, tool_call.data['id'])
        
        if reminder.user != user:
            return ascog.ToolAnswerMessage({'error': 'Vous n\'êtes pas l\'auteur de ce rappel.'}, tool_call.data['id'])
        
        self.remove_reminder(reminder_id)
        return ascog.ToolAnswerMessage({'user': user.id, 'reminder_id': reminder_id}, tool_call.data['id'])
    
    
async def setup(bot):
    await bot.add_cog(Reminders(bot))
